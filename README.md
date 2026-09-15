# ComfyUI-DJ_ApilioAPI

> 版本 `260915-224049` · 作者：大江

ComfyUI 自定义节点，调用 Apilio 图像 API 做文生图 / 图生图，内置比例保险与失败兜底。

## ✨ 功能

- 节点 `DJ_Apilio_Image_v1_1`（显示名 `Apilio 图像生成 v1.1`），分类 `DJ_ApilioAPI`
- 支持模型：`gpt-image-2`、`gpt-image-2.5-sunburst`（按 2K/4K 自动拼后缀 `-2k` / `-4k`）
- 尺寸档位：`2K` / `4K`；比例：`auto` 或 10 种固定比例（1:1、2:3、3:2、3:4、4:3、4:5、5:4、9:16、16:9、21:9）
- 最多 10 张参考图（`image_1` ~ `image_10`），有参考图走 `/v1/images/edits`，否则走 `/v1/images/generations`
- 比例保险：后端返回比例/尺寸不符时自动居中裁剪 + 双线性缩放到目标尺寸
- 失败兜底：API 失败时输出一张带错误信息的占位图，不中断工作流
- 自动重试：最多 3 次，指数退避（2s / 4s / 8s，上限 30s）
- 老工作流兼容：`VALIDATE_INPUTS` 放行已停用的 `1K` 档位，运行时自动回落到 `2K`

## 安装

把 `ComfyUI-DJ_ApilioAPI` 文件夹放进 ComfyUI 的 `custom_nodes\` 目录，重启 ComfyUI。

## 依赖

- `numpy`
- `requests`
- `torch`
- `Pillow`
- `urllib3`

## 作者

大江
