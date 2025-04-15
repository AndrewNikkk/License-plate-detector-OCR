import cv2
import numpy as np
from paddleocr import PaddleOCR


def enhance_text_ocr(
    image,
    scale_factor=2,
    apply_binarization=True,
    use_clahe=True,
    use_deconvolution=True,
    use_sharpening=False,
    use_morphology=True,
    clahe_params={'clip_limit': 3.0, 'tile_size': (8, 8)},
    deconv_kernel_size=(5, 5),
    sharpening_params={'amount': 1.5, 'kernel_size': (5, 5)},
    morph_params={'kernel_size': (3, 3), 'op': 'close', 'kernel_type': 'ellipse'}
):
    """
    Полная предобработка изображения для улучшения распознавания текста
    
    Параметры:
        image: Входное изображение в формате BGR
        scale_factor: Коэффициент увеличения изображения
        apply_binarization: Применять ли бинаризацию Оцу
        use_clahe: Использовать CLAHE для улучшения контраста
        use_deconvolution: Применять деконволюцию
        use_sharpening: Применять unsharp mask
        use_morphology: Применять морфологические операции
        clahe_params: Параметры для CLAHE {'clip_limit': float, 'tile_size': tuple}
        deconv_kernel_size: Размер ядра для деконволюции
        sharpening_params: Параметры резкости {'amount': float, 'kernel_size': tuple}
        morph_params: Параметры морфологии {'kernel_size': tuple, 'op': str, 'kernel_type': str}
    
    Возвращает:
        Обработанное изображение в формате RGB
    """
    
    scaled = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_params['clip_limit'],
            tileGridSize=clahe_params['tile_size']
        )
        l = clahe.apply(l)
    
    if use_deconvolution:
        kernel = np.ones(deconv_kernel_size, np.float32) / (deconv_kernel_size[0] * deconv_kernel_size[1])
        l = cv2.filter2D(l, -1, kernel)
    
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    if use_sharpening:
        blurred = cv2.GaussianBlur(enhanced_bgr, sharpening_params['kernel_size'], 0)
        enhanced_bgr = cv2.addWeighted(
            enhanced_bgr,
            1.0 + sharpening_params['amount'],
            blurred,
            -sharpening_params['amount'],
            0
        )
        enhanced_bgr = np.clip(enhanced_bgr, 0, 255).astype(np.uint8)
    
    if apply_binarization:
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = binary
    else:
        result = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    
    if use_morphology:
        kernel_type = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }.get(morph_params['kernel_type'].lower(), cv2.MORPH_ELLIPSE)
        
        kernel = cv2.getStructuringElement(
            kernel_type,
            morph_params['kernel_size']
        )
        
        if morph_params['op'] == 'close':
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif morph_params['op'] == 'open':
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif morph_params['op'] == 'both':
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result