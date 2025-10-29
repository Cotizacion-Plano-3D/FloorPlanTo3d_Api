
import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# CONFIG (ajusta si quieres)
# ---------------------------------------------------------------------------
DOOR_REAL_WIDTH_M = 0.90  # ancho típico de puerta para calibración por defecto
FLOOR_HEIGHT_M = float(os.environ.get("FLOOR_HEIGHT_M", "2.5"))

# Umbrales mínimos de área (en pixeles) para filtrar ruido por clase
MIN_PX_AREA = {
    "wall": 400,
    "window": 50,
    "door": 100,
}

# ---------------------------------------------------------------------------
# UTILIDADES DE ESCALA
# ---------------------------------------------------------------------------
def door_scale_from_bboxes(rois: np.ndarray, class_ids: np.ndarray) -> Optional[float]:
    """
    Retorna la mediana del lado largo de las puertas en pixeles.
    rois: [N, (y1,x1,y2,x2)]
    class_ids: [N]
    """
    long_sides = []
    for bb, cid in zip(rois, class_ids):
        if cid != 3:  # 3 = door
            continue
        wpx = int(bb[3] - bb[1])
        hpx = int(bb[2] - bb[0])
        long_sides.append(max(wpx, hpx))
    if not long_sides:
        return None
    long_sides.sort()
    return float(long_sides[len(long_sides)//2])

def compute_scale_factor(r: Dict[str, Any], fallback: float = 0.01) -> float:
    door_px = door_scale_from_bboxes(r.get('rois', np.zeros((0,4))), r.get('class_ids', np.zeros((0,), dtype=int)))
    if door_px and door_px > 0:
        return DOOR_REAL_WIDTH_M / door_px
    return fallback

# ---------------------------------------------------------------------------
# FILTRADO DE DETECCIONES
# ---------------------------------------------------------------------------
def filter_detections(result: Dict[str, Any],
                      min_score: float = 0.5,
                      min_px_area: Dict[str, int] = MIN_PX_AREA) -> Dict[str, Any]:
    """
    Filtra detecciones por score y área mínima de máscara (en píxeles).
    Clases esperadas: 1=wall, 2=window, 3=door (ajusta si tu modelo difiere).
    """
    if result is None or 'class_ids' not in result or result['class_ids'] is None:
        return result

    keep: List[int] = []
    class_ids = result['class_ids']
    scores = result.get('scores', np.ones_like(class_ids, dtype=float))
    masks = result.get('masks', None)

    def cname(cid: int) -> str:
        return {1: "wall", 2: "window", 3: "door"}.get(int(cid), f"class_{cid}")

    for i, cid in enumerate(class_ids):
        score = float(scores[i]) if scores is not None and len(scores) > i else 1.0
        if masks is not None and masks.size > 0 and masks.ndim == 3 and masks.shape[-1] > i:
            area = int(masks[..., i].sum())
        else:
            area = 0

        name = cname(int(cid))
        if score >= min_score and area >= min_px_area.get(name, 0):
            keep.append(i)

    if not keep:
        # nada pasa el filtro; retornamos tal cual para no romper flujo
        return result

    result['class_ids'] = result['class_ids'][keep]
    if 'scores' in result and result['scores'] is not None:
        result['scores'] = result['scores'][keep]
    if 'rois' in result and result['rois'] is not None and len(result['rois']) > 0:
        result['rois'] = result['rois'][keep]
    if 'masks' in result and result['masks'] is not None and result['masks'].ndim == 3:
        result['masks'] = result['masks'][..., keep]
    return result

# ---------------------------------------------------------------------------
# MÁSCARAS Y CONTORNOS (VERSIÓN ADAPTATIVA)
# ---------------------------------------------------------------------------
def _wall_union_mask(masks: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Une todas las máscaras de muros (class_id == 1) en 0/255 uint8."""
    if masks is None or masks.size == 0 or class_ids is None or len(class_ids) == 0:
        return np.zeros((1,1), np.uint8)
    wall_idx = [i for i, c in enumerate(class_ids) if int(c) == 1]
    if not wall_idx:
        return np.zeros(masks.shape[:2], np.uint8)
    wall_union = np.any([masks[..., i] for i in wall_idx], axis=0).astype(np.uint8) * 255
    return wall_union

def _adaptive_close(binary: np.ndarray, max_iters: int = 4, base_kernel_ratio: float = 0.004) -> np.ndarray:
    """
    Cierra brechas finas entre muros de manera adaptativa.
    - base_kernel_ratio * max(H, W) -> tamaño inicial de kernel (impar).
    - Itera cierre/dilatación + open leve hasta estabilizar el área.
    """
    h, w = binary.shape[:2]
    k0 = max(3, int(round(max(h, w) * base_kernel_ratio)) | 1)  # impar y >=3
    img = binary.copy()
    last_area = 0.0

    for it in range(max_iters):
        k = k0 + 2 * it  # 3,5,7,9...
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        dil = cv2.dilate(closed, kernel, iterations=1)
        opened = cv2.morphologyEx(dil, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            img = opened
            continue
        main = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(main))
        if last_area > 0 and abs(area - last_area) / max(last_area, 1e-6) < 0.01:
            return opened
        last_area = area
        img = opened

    return img

def _main_contour_and_holes(binary: np.ndarray, min_hole_ratio: float = 0.01):
    """
    Devuelve el contorno exterior principal y huecos relevantes.
    Usa CCOMP para distinguir huecos y descarta huecos pequeños (< min_hole_ratio).
    """
    bin255 = (binary > 0).astype(np.uint8) * 255

    cnts_ext, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_ext:
        return None, []

    main = max(cnts_ext, key=cv2.contourArea)
    ext_area = float(cv2.contourArea(main))

    cnts_all, hier = cv2.findContours(bin255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is not None and len(cnts_all) > 0:
        for i, h in enumerate(hier[0]):
            parent = int(h[3])
            if parent != -1:  # hijo => posible hueco
                a = float(cv2.contourArea(cnts_all[i]))
                if ext_area > 0 and (a / ext_area) >= min_hole_ratio:
                    holes.append(cnts_all[i])

    return main, holes

def built_area_m2_from_contour_adaptive(masks: np.ndarray, class_ids: np.ndarray, scale_factor: float,
                                        base_kernel_ratio: float = 0.004,
                                        max_iters: int = 4,
                                        subtract_holes: bool = True,
                                        min_hole_ratio: float = 0.01) -> float:
    """Área techada aproximada en m² (contorno exterior - huecos grandes)."""
    wall_union = _wall_union_mask(masks, class_ids)
    if wall_union.max() == 0:
        return 0.0

    wall_union = cv2.medianBlur(wall_union, 3)
    closed = _adaptive_close(wall_union, max_iters=max_iters, base_kernel_ratio=base_kernel_ratio)

    main, holes = _main_contour_and_holes(closed, min_hole_ratio=min_hole_ratio)
    if main is None:
        return 0.0

    area_px = float(cv2.contourArea(main))
    if subtract_holes and holes:
        holes_area_px = sum(float(cv2.contourArea(h)) for h in holes)
        area_px -= holes_area_px
    return float(area_px) * (scale_factor ** 2)

def outer_perimeter_m_from_walls_adaptive(masks: np.ndarray, class_ids: np.ndarray, scale_factor: float,
                                          base_kernel_ratio: float = 0.004,
                                          max_iters: int = 4) -> float:
    """Perímetro exterior (m) del contorno principal cerrado."""
    wall_union = _wall_union_mask(masks, class_ids)
    if wall_union.max() == 0:
        return 0.0

    wall_union = cv2.medianBlur(wall_union, 3)
    closed = _adaptive_close(wall_union, max_iters=max_iters, base_kernel_ratio=base_kernel_ratio)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    main = max(cnts, key=cv2.contourArea)
    perim_px = float(cv2.arcLength(main, True))
    return perim_px * scale_factor

# ---------------------------------------------------------------------------
# MÉTRICAS CON MÁSCARAS
# ---------------------------------------------------------------------------
def area_m2_from_mask(mask: np.ndarray, scale_factor: float) -> float:
    px = int(mask.sum())
    return float(px) * (scale_factor ** 2)

def wall_length_m_from_mask(mask: np.ndarray, scale_factor: float) -> float:
    sk = skeletonize(mask.astype(bool))
    px_len = int(np.count_nonzero(sk))
    return float(px_len) * scale_factor

def calcular_medidas_extraidas(result: Dict[str, Any],
                               scale_factor: float,
                               floor_height_m: float = FLOOR_HEIGHT_M) -> Dict[str, Any]:
    """
    Calcula métricas clave a partir de masks/class_ids con escala.
    Campos devueltos compatibles con tu salida previa.
    """
    masks = result.get('masks', None)
    class_ids = result.get('class_ids', None)

    area_paredes_m2 = 0.0
    area_ventanas_m2 = 0.0
    wall_axis_length_m = 0.0

    if masks is not None and masks.size > 0 and class_ids is not None:
        for i in range(masks.shape[-1]):
            m = masks[..., i]
            cid = int(class_ids[i])
            if cid == 1:  # wall
                L = wall_length_m_from_mask(m, scale_factor)
                wall_axis_length_m += L
                area_paredes_m2 += L * floor_height_m  # una cara
            elif cid == 2:  # window
                area_ventanas_m2 += area_m2_from_mask(m, scale_factor)

    # área y perímetro EXTERIORES (mejorados con cierre adaptativo + huecos)
    area_total_m2 = built_area_m2_from_contour_adaptive(
        masks, class_ids, scale_factor,
        base_kernel_ratio=0.004, max_iters=4,
        subtract_holes=True, min_hole_ratio=0.01
    )
    perimetro_total_m = outer_perimeter_m_from_walls_adaptive(
        masks, class_ids, scale_factor,
        base_kernel_ratio=0.004, max_iters=4
    )

    return {
        "area_paredes_m2": round(area_paredes_m2, 2),
        "area_total_m2": round(area_total_m2, 2),
        "area_ventanas_m2": round(area_ventanas_m2, 2),
        "perimetro_total_m": round(perimetro_total_m, 2),
        "wall_axis_length_m": round(wall_axis_length_m, 2)
    }

# ---------------------------------------------------------------------------
# EJEMPLO DE INTEGRACIÓN (simulado)
# ---------------------------------------------------------------------------
def postprocess_detection_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejemplo de integración:
    - filtra detecciones
    - calcula escala
    - computa métricas mejoradas
    Retorna un dict con "medidas_extraidas"
    """
    r = filter_detections(raw_result, min_score=0.5)
    scale_factor = compute_scale_factor(r, fallback=0.01)

    medidas = calcular_medidas_extraidas(r, scale_factor, floor_height_m=FLOOR_HEIGHT_M)

    # agrega escala y conteos para compatibilidad
    class_ids = r.get('class_ids')
    medidas.update({
        "escala_calculada": round(scale_factor, 4),
        "num_paredes": int((class_ids == 1).sum()) if class_ids is not None else 0,
        "num_puertas": int((class_ids == 3).sum()) if class_ids is not None else 0,
        "num_ventanas": int((class_ids == 2).sum()) if class_ids is not None else 0,
    })

    return {"medidas_extraidas": medidas}


# ---------------------------------------------------------------------------
# Si este archivo se ejecuta solo, hace una prueba "mock"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # MOCK: estructura mínima esperada del detector
    H, W = 800, 1200
    # Creamos máscaras vacías para demo
    masks = np.zeros((H, W, 0), dtype=np.uint8)
    class_ids = np.zeros((0,), dtype=int)
    rois = np.zeros((0,4), dtype=int)
    scores = np.zeros((0,), dtype=float)

    demo = {"masks": masks, "class_ids": class_ids, "rois": rois, "scores": scores}
    out = postprocess_detection_output(demo)
    print(json.dumps(out, indent=2, ensure_ascii=False))
