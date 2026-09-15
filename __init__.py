from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import comfy.model_management


class ConsoleColor:
    VIOLET = '\033[38;5;135m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def get_display_width(text):
    return sum(2 if ord(char) > 0x7F else 1 for char in text)


def log_custom(icon, msg, task_id="T-00", color=ConsoleColor.CYAN):
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{task_id}] {color}{icon} {msg}{ConsoleColor.END}")


def print_header(model, size, ratio):
    title = f" {model} 开始生成 ({size} | {ratio}) "
    border = "═" * get_display_width(title)
    print(f"\n{ConsoleColor.VIOLET}╔{border}╗\n║{title}║\n╚{border}╝{ConsoleColor.END}")


def draw_error_image(msg_raw, width=1024, height=1024):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_path = "C:\\Windows\\Fonts\\msyh.ttc" if os.name == "nt" else "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        f_title = ImageFont.truetype(font_path, 50)
        f_main = ImageFont.truetype(font_path, 30)
    except Exception:
        f_title = f_main = ImageFont.load_default()

    draw.rectangle([0, 0, width, 140], fill=(255, 235, 235))
    draw.text((50, 40), "生成失败", fill=(200, 40, 40), font=f_title)
    wrapped = "\n".join(msg_raw[i:i + 45] for i in range(0, len(msg_raw), 45))
    draw.multiline_text((50, 200), f"未生成图像\n\n原因详情：\n{wrapped}", fill=(60, 60, 60), font=f_main, spacing=15)
    return pil2tensor(img)


def extract_images_from_response(resp):
    try:
        data = resp.json()
    except Exception:
        return [], "", resp.text

    b64_srcs = []
    url_srcs = []

    def walk(val):
        if not val:
            return
        if isinstance(val, str):
            s = val.strip()
            if "![" in s and "](" in s:
                url_srcs.extend(re.findall(r"!\[.*?\]\((https?://[^\)\s]+)\)", s))
                url_srcs.extend(re.findall(r"src=[\"'](https?://[^\"']+)[\"']", s))
                return
            if s.startswith("http") and " " not in s and "\n" not in s:
                url_srcs.append(s)
            elif s.startswith("data:image/"):
                b64_srcs.append(s.split(",", 1)[-1])
            return
        if isinstance(val, list):
            for item in val:
                walk(item)
            return
        if isinstance(val, dict):
            inline_data = val.get("inlineData") or val.get("inline_data")
            if isinstance(inline_data, dict) and inline_data.get("data"):
                b64_srcs.append(inline_data["data"])
            if val.get("b64_json"):
                b64_srcs.append(val["b64_json"])
            for key in ("url", "imageUrl", "image_url"):
                value = val.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    url_srcs.append(value)
            for value in val.values():
                walk(value)

    walk(data)
    return list(dict.fromkeys(b64_srcs)), next(iter(dict.fromkeys(url_srcs)), ""), json.dumps(data, ensure_ascii=False)


class ApilioImageNode:
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate_images"
    OUTPUT_NODE = True
    CATEGORY = "DJ_ApilioAPI"

    NO_PROXY = {"http": None, "https": None}

    _SIZE_MAP = {
        ("1:1", "2K"): "2048x2048", ("1:1", "4K"): "4096x4096",
        ("2:3", "2K"): "1365x2048", ("2:3", "4K"): "2730x4096",
        ("3:2", "2K"): "2048x1365", ("3:2", "4K"): "4096x2730",
        ("3:4", "2K"): "1728x2304", ("3:4", "4K"): "3072x4096",
        ("4:3", "2K"): "2304x1728", ("4:3", "4K"): "4096x3072",
        ("4:5", "2K"): "1638x2048", ("4:5", "4K"): "3276x4096",
        ("5:4", "2K"): "2048x1638", ("5:4", "4K"): "4096x3276",
        ("9:16", "2K"): "1440x2560", ("9:16", "4K"): "2160x3840",
        ("16:9", "2K"): "2560x1440", ("16:9", "4K"): "3840x2160",
        ("21:9", "2K"): "3024x1296", ("21:9", "4K"): "4096x1728",
    }

    _SIZE_CHOICES = ("2K", "4K")

    _MODEL_MAPPING = {
        "gpt-image-2": "gpt-image-2",
        "gpt-image-2.5-sunburst": "gpt-image-2.5-sunburst",
    }

    _RATIO_FLOATS = {
        "1:1": 1.0, "2:3": 2 / 3, "3:2": 3 / 2,
        "3:4": 3 / 4, "4:3": 4 / 3, "4:5": 4 / 5,
        "5:4": 5 / 4, "9:16": 9 / 16, "16:9": 16 / 9,
        "21:9": 21 / 9,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model_type": (["gpt-image-2", "gpt-image-2.5-sunburst"], {"default": "gpt-image-2"}),
                "image_size": (["2K", "4K"], {"default": "2K"}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "生成后控制": (["randomize", "fixed"], {"default": "randomize"}),
                **{f"image_{i}": ("IMAGE",) for i in range(1, 11)},
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image_size):
        # 只接管 image_size 的校验：老工作流里存的 "1K" 直接放行，
        # 由 generate_images 兜底纠正，不弹红框报错。
        return True

    def generate_images(self, prompt, api_key, model_type, image_size, aspect_ratio, seed=-1, 生成后控制="randomize", **kwargs):
        start_time = time.time()
        tid = f"T-{random.randint(10, 99)}"
        api_key = api_key.strip()

        if image_size not in self._SIZE_CHOICES:
            log_custom(
                "⚠",
                f"尺寸档位 {image_size!r} 已停用（现只有 2K / 4K），本次自动改用 {self._SIZE_CHOICES[0]}",
                tid,
                ConsoleColor.YELLOW,
            )
            image_size = self._SIZE_CHOICES[0]

        if not api_key:
            return (draw_error_image("未填写 API Key"), "", "failed", "Missing API Key")

        input_images = self._collect_images(kwargs)
        final_ratio = self._resolve_ratio(aspect_ratio, input_images)
        curr_seed = seed if seed != -1 else random.randint(1, 0xffffffffffffffff)
        actual_model = self._resolve_model(model_type, image_size)

        print_header(actual_model, image_size, final_ratio)

        try:
            result = self._run_apilio(api_key, prompt, actual_model, image_size, final_ratio, curr_seed, input_images, tid)
        except Exception as exc:
            err = str(exc)
            log_custom("✗", f"生成失败: {err}", tid, ConsoleColor.RED)
            return (draw_error_image(err), "", "failed", err)

        elapsed = round(time.time() - start_time, 1)
        response = f"### 生成成功\n- 模型: `{actual_model}`\n- 比例: `{final_ratio}`\n- 耗时: `{elapsed}s`"

        return (result["tensor"], result.get("url", ""), str(curr_seed), response)

    def _collect_images(self, kwargs):
        images = []
        for i in range(1, 11):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                images.append(img)
        return images

    def _resolve_ratio(self, aspect_ratio, images):
        if aspect_ratio != "auto":
            return aspect_ratio
        if not images:
            return "1:1"
        h, w = images[0].shape[1], images[0].shape[2]
        value = w / h
        ratios = {
            1.0: "1:1", 0.666: "2:3", 1.5: "3:2", 0.75: "3:4", 1.333: "4:3",
            0.8: "4:5", 1.25: "5:4", 0.5625: "9:16", 1.777: "16:9", 2.333: "21:9",
        }
        return ratios[min(ratios.keys(), key=lambda x: abs(x - value))]

    def _resolve_model(self, model_type, image_size):
        base_name = self._MODEL_MAPPING.get(model_type, model_type)
        suffix = {"2K": "-2k", "4K": "-4k"}.get(image_size, "")
        return f"{base_name}{suffix}"

    def _image_size(self, ratio, image_size):
        return self._SIZE_MAP.get((ratio, image_size), self._SIZE_MAP[("1:1", image_size)])

    def _images_to_b64(self, images):
        encoded = []
        for image in images:
            buf = BytesIO()
            tensor2pil(image).save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return encoded

    def _prompt_with_size_hint(self, prompt, image_size, ratio):
        final_prompt = prompt.strip() or "a beautiful image"
        hints = []
        if image_size:
            hints.append(f"分辨率: {image_size}")
        if ratio:
            hints.append(f"比例: {ratio}")
        if hints:
            final_prompt = f"{final_prompt} [{', '.join(hints)}]"
        if ratio and ratio not in ("auto", "1:1"):
            final_prompt = (
                f"{final_prompt}\n\n"
                f"画幅比例硬性要求：最终图片必须严格输出 {ratio} 画幅；"
                "不要输出 1:1 正方形，不要在正方形画布内留白。"
            )
        return final_prompt

    @staticmethod
    def _parse_size(size_text):
        match = re.match(r"^(\d+)x(\d+)$", str(size_text or ""))
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))

    def _force_result_aspect_ratio(self, result, ratio, image_size, tid, allow_upscale=True):
        if not result or ratio in ("", "auto", "1:1"):
            return result
        tensor = result.get("tensor")
        if tensor is None or tensor.ndim != 4:
            return result

        target_size = self._image_size(ratio, image_size)
        target_w, target_h = self._parse_size(target_size)
        target_ratio = self._RATIO_FLOATS.get(ratio)
        if not target_ratio and target_w and target_h:
            target_ratio = target_w / target_h
        if not target_ratio:
            return result

        batch, height, width, channels = tensor.shape
        if height <= 0 or width <= 0:
            return result

        if not allow_upscale and target_w and target_h and (width < target_w or height < target_h):
            log_custom(
                "⚠",
                f"后端返回尺寸 {width}x{height} 低于请求 {target_w}x{target_h}，保留原图，不执行放大",
                tid,
                ConsoleColor.RED,
            )
            return result

        current_ratio = width / height
        needs_resize = target_w and target_h and (width != target_w or height != target_h)
        if abs(current_ratio - target_ratio) < 0.01 and not needs_resize:
            return result

        fixed = tensor
        if current_ratio > target_ratio:
            new_width = max(1, min(width, int(round(height * target_ratio))))
            left = max(0, (width - new_width) // 2)
            fixed = fixed[:, :, left:left + new_width, :]
        else:
            new_height = max(1, min(height, int(round(width / target_ratio))))
            top = max(0, (height - new_height) // 2)
            fixed = fixed[:, top:top + new_height, :, :]

        if target_w and target_h:
            nchw = fixed.permute(0, 3, 1, 2)
            nchw = F.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
            fixed = nchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        result["tensor"] = fixed
        log_custom("▣", f"比例保险生效: {width}x{height} -> {fixed.shape[2]}x{fixed.shape[1]} ({ratio})", tid, ConsoleColor.YELLOW)
        return result

    def _run_apilio(self, api_key, prompt, model, image_size, ratio, seed, images, tid):
        base_url = "https://api.apilio.ai"
        clean_model = model.split("/")[-1]
        target_size = self._image_size(ratio, image_size)
        prompt_text = self._prompt_with_size_hint(prompt, image_size, ratio)

        payload = {
            "model": clean_model,
            "prompt": prompt_text,
            "size": target_size,
            "n": 1,
            "response_format": "url",
            "aspect_ratio": ratio,
        }
        if seed != -1:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Return-Base64": "1",
            "X-No-CDN": "1",
        }

        max_tries = 3
        last_err = None

        for attempt in range(max_tries):
            comfy.model_management.throw_exception_if_processing_interrupted()
            endpoint_name = "edits" if images else "generations"
            log_custom(
                "→",
                f"调用 Apilio API /{endpoint_name} | {target_size} ({attempt + 1}/{max_tries})",
                tid,
                ConsoleColor.BLUE,
            )

            try:
                if images:
                    files = []
                    try:
                        for idx, image in enumerate(images):
                            buf = BytesIO()
                            tensor2pil(image).save(buf, format="PNG")
                            buf.seek(0)
                            files.append(("image", (f"image_{idx + 1}.png", buf, "image/png")))
                        resp = requests.post(
                            f"{base_url}/v1/images/edits",
                            headers={"Authorization": f"Bearer {api_key}"},
                            data={key: str(value) for key, value in payload.items()},
                            files=files,
                            timeout=900,
                            verify=False,
                            proxies=self.NO_PROXY,
                        )
                    finally:
                        for _, file_tuple in files:
                            file_tuple[1].close()
                else:
                    resp = requests.post(
                        f"{base_url}/v1/images/generations",
                        headers=headers,
                        json=payload,
                        timeout=900,
                        verify=False,
                        proxies=self.NO_PROXY,
                    )
            except requests.exceptions.RequestException as exc:
                last_err = f"网络错误: {exc}"
            else:
                if resp.status_code in (401, 402, 403):
                    raise RuntimeError(f"鉴权/余额失败 ({resp.status_code}): {resp.text[:500]}")
                if resp.status_code == 200:
                    parsed = self._decode_response_images(resp)
                    if parsed:
                        return self._force_result_aspect_ratio(
                            parsed, ratio, image_size, tid, allow_upscale=False
                        )
                    last_err = f"返回 200 但未解析到图片。响应前 200 字: {resp.text[:200]}"
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"

            wait = min((2 ** attempt) * 2, 30)
            log_custom("…", f"{wait}s 后重试", tid, ConsoleColor.YELLOW)
            time.sleep(wait)

        raise RuntimeError(last_err or "未知错误")

    def _decode_response_images(self, resp):
        b64s, url, _ = extract_images_from_response(resp)
        if b64s:
            tensors = [pil2tensor(Image.open(BytesIO(base64.b64decode(b64)))) for b64 in b64s]
            log_custom("✓", f"出图成功 ({len(tensors)} 张)", color=ConsoleColor.GREEN)
            return {"tensor": torch.cat(tensors, dim=0), "url": url, "count": len(tensors)}
        if url:
            img_bin = requests.get(url, timeout=30, verify=False, proxies=self.NO_PROXY).content
            img_pil = Image.open(BytesIO(img_bin))
            log_custom("✓", f"URL 下载成功 {img_pil.size}", color=ConsoleColor.GREEN)
            return {"tensor": pil2tensor(img_pil), "url": url, "count": 1}
        return None


NODE_CLASS_MAPPINGS = {
    "DJ_Apilio_Image_v1_1": ApilioImageNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DJ_Apilio_Image_v1_1": "Apilio 图像生成 v1.1"
}
