import sys  # 系统相关模块
import os

# 设置控制台编码
if sys.platform == 'win32':  # windows平台的标识
    os.system('chcp 65001 > nul')  # utf-8
elif sys.platform == 'darwin':  # macOS
    pass  # 默认使用utf-8,无需额外设置

import cv2
# pip install opencv-python
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import time
import logging
from pathlib import Path  # 路径处理模块
import glob  # 用于查找制定格式的图片文件
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleBatchOCR:
    """简化版批量OCR处理器"""

    def __init__(self, input_folder="pictures", output_folder="output_pictures"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.ocr_engines = {}
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # 创建输出文件夹
        self.output_folder.mkdir(exist_ok=True)

        # 初始化OCR引擎
        self._init_ocr_engines()

    def _init_ocr_engines(self):  # 私有化方法
        """初始化OCR引擎"""
        logger.info("正在初始化OCR引擎...")

        # 1. 尝试Tesseract
        try:
            import pytesseract

            # 尝试多个可能的Tesseract路径
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\peng.cheng\AppData\Local\miniforge3\Library\bin\tesseract.exe',
                'tesseract'  # 系统PATH中
            ]

            tesseract_found = False
            for path in tesseract_paths:
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    version = pytesseract.get_tesseract_version()
                    tesseract_found = True
                    logger.info(f"✅ 找到Tesseract {version}: {path}")
                    break
                except:
                    continue

            if tesseract_found:
                self.ocr_engines['tesseract'] = pytesseract
                logger.info("✅ Tesseract初始化成功")
            else:
                logger.error("❌ 无法找到Tesseract可执行文件")

        except Exception as e:
            logger.error(f"❌ Tesseract初始化失败: {e}")

        # 2. 添加OpenCV简单文字检测
        self.ocr_engines['opencv_simple'] = "opencv_fallback"
        logger.info("✅ OpenCV简单OCR初始化成功")

        if self.ocr_engines:
            available_engines = list(self.ocr_engines.keys())
            logger.info(f"✅ 可用OCR引擎: {', '.join(available_engines)}")
        else:
            logger.error("❌ 没有可用的OCR引擎")

    def get_image_files(self):
        """获取输入文件夹中的所有图片文件"""
        image_files = []

        if not self.input_folder.exists():
            logger.error(f"输入文件夹不存在: {self.input_folder}")
            return image_files

        for ext in self.supported_formats:
            pattern = str(self.input_folder / f"*{ext}")  # 路径拼接
            files = glob.glob(pattern, recursive=False)  # pattern表示搜索
            image_files.extend(files)

            # 也搜索大写扩展名
            pattern = str(self.input_folder / f"*{ext.upper()}")
            files = glob.glob(pattern, recursive=False)
            image_files.extend(files)

        # 去重并排序
        image_files = sorted(list(set(image_files)))
        logger.info(f"找到 {len(image_files)} 张图片文件")

        for img_file in image_files:
            logger.info(f"  - {os.path.basename(img_file)}")

        return image_files

    def preprocess_image(self, image_path):
        """图像预处理"""
        try:
            # 读取图像
            image = cv2.imread(image_path)  # 返回图像数据（数组的形式）
            if image is None:  # 检查图像是否成功读取
                logger.error(f"无法读取图像: {image_path}")
                return None

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR表示彩色 GRAY灰度， 由A-B

            # 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            return cleaned
        except Exception as e:
            logger.error(f"图像预处理失败 {image_path}: {e}")
            return None

    def ocr_with_tesseract(self, image_path):
        """使用Tesseract进行识别"""
        if 'tesseract' not in self.ocr_engines:
            return None

        try:
            import pytesseract

            # 读取图像
            image = Image.open(image_path)

            # 尝试多种语言配置
            lang_configs = [
                'chi_sim+eng',  # 中文简体+英文
                'chi_sim',  # 仅中文简体
                'eng',  # 仅英文
            ]

            best_result = ""
            best_confidence = 0

            for lang in lang_configs:
                try:
                    # 获取文本和置信度
                    text = pytesseract.image_to_string(image, lang=lang).strip()  # 去除收尾空白的字符
                    logger.debug(f"语言 {lang} 识别文本: '{text}'")  # 调试阶段性结果输出

                    # 获取详细信息（包括置信度）
                    data = pytesseract.image_to_data(image, lang=lang,
                                                     output_type=pytesseract.Output.DICT)  # 输出结果的格式，就是包含置信度等详细信息的字典

                    # 计算平均置信度
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    logger.debug(f"语言 {lang} 平均置信度: {avg_confidence:.1f}%")

                    # 如果有文本且置信度更高，则更新最佳结果
                    if text and (avg_confidence > best_confidence or not best_result):
                        best_result = text
                        best_confidence = avg_confidence
                        logger.debug(f"更新最佳结果: '{text}' (置信度: {avg_confidence:.1f}%)")

                except Exception as e:
                    logger.error(f"语言 {lang} 识别失败: {e}")
                    continue

            return {
                'engine': 'Tesseract',
                'text': best_result,
                'confidence': best_confidence,
                'success': len(best_result) > 0 and best_confidence > 10
            }

        except Exception as e:
            logger.error(f"Tesseract识别失败 {image_path}: {e}")
            return None

    def ocr_with_opencv_simple(self, image_path):
        """使用OpenCV简单文字检测"""
        try:
            # 预处理图像
            binary_image = self.preprocess_image(image_path)
            if binary_image is None:
                return None

            # 查找轮廓。边缘工程的一部分
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 统计文字区域
            text_regions = []
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 10:  # 过滤小区域
                    area = cv2.contourArea(contour)
                    if area > 100:  # 面积过滤
                        aspect_ratio = w / h
                        if 0.1 < aspect_ratio < 10:  # 长宽比过滤
                            text_regions.append({
                                'id': i,
                                'bbox': (x, y, w, h),
                                'area': area,
                                'aspect_ratio': aspect_ratio
                            })

            result_text = f"检测到 {len(text_regions)} 个文字区域"

            return {
                'engine': 'OpenCV简单检测',
                'text': result_text,
                'regions': text_regions,
                'success': len(text_regions) > 0
            }
        except Exception as e:
            logger.error(f"OpenCV检测失败 {image_path}: {e}")
            return None

    def create_visualization(self, image_path, ocr_results, output_path):
        """创建可视化结果"""
        try:
            # 读取原始图像
            image = cv2.imread(image_path)
            if image is None:
                return False

            # 如果有OpenCV检测结果，绘制边界框
            opencv_result = ocr_results.get('opencv_simple')
            if opencv_result and 'regions' in opencv_result:
                for region in opencv_result['regions']:
                    x, y, w, h = region['bbox']
                    # 绘制矩形框
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 添加标签
                    cv2.putText(image, f"R{region['id']}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 如果有Tesseract结果，添加文本信息
            tesseract_result = ocr_results.get('tesseract')
            if tesseract_result and tesseract_result['success']:
                # 在图像顶部添加识别的文本
                text = tesseract_result['text'][:50] + "..." if len(tesseract_result['text']) > 50 else \
                    tesseract_result['text']
                cv2.putText(image, f"Text: {text}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f"Confidence: {tesseract_result['confidence']:.1f}%",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 保存可视化结果
            cv2.imwrite(str(output_path), image)
            return True
        except Exception as e:
            logger.error(f"创建可视化失败: {e}")
            return False

    def process_single_image(self, image_path):
        """处理单张图片"""
        image_name = os.path.basename(image_path)
        logger.info(f"处理图片: {image_name}")

        start_time = time.time()
        results = {}

        # 使用Tesseract识别
        if 'tesseract' in self.ocr_engines:
            logger.info("  使用Tesseract识别...")
            result = self.ocr_with_tesseract(image_path)
            if result:
                results['tesseract'] = result
                if result['success']:
                    logger.info(f"    识别成功: {result['text'][:50]}...")
                    logger.info(f"    置信度: {result['confidence']:.1f}%")
                else:
                    logger.info("    识别失败或置信度过低")

        # 使用OpenCV检测
        if 'opencv_simple' in self.ocr_engines:
            logger.info("  使用OpenCV检测...")
            result = self.ocr_with_opencv_simple(image_path)
            if result:
                results['opencv_simple'] = result
                logger.info(f"    结果: {result['text']}")

        processing_time = time.time() - start_time

        # 生成输出文件名
        base_name = Path(image_name).stem

        # 保存JSON结果
        json_output = self.output_folder / f"{base_name}_ocr_result.json"
        result_data = {
            'image_path': image_path,
            'image_name': image_name,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'ocr_results': results
        }

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

        # 保存文本结果
        txt_output = self.output_folder / f"{base_name}_ocr_text.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(f"图片: {image_name}\n")
            f.write(f"处理时间: {processing_time:.2f}秒\n")
            f.write(f"处理时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            for engine, result in results.items():
                f.write(f"【{result['engine']}】\n")
                f.write(f"识别文本: {result['text']}\n")
                f.write(f"识别成功: {'是' if result['success'] else '否'}\n")
                if 'confidence' in result:
                    f.write(f"置信度: {result['confidence']:.1f}%\n")
                f.write("-" * 30 + "\n")

        # 创建可视化图像
        vis_output = self.output_folder / f"{base_name}_visualization.jpg"
        self.create_visualization(image_path, results, vis_output)

        logger.info(f"  ✅ 处理完成，耗时 {processing_time:.2f}秒")
        logger.info(f"    JSON结果: {json_output.name}")
        logger.info(f"    文本结果: {txt_output.name}")
        logger.info(f"    可视化图: {vis_output.name}")

        return result_data

    def process_all_images(self):
        """批量处理所有图片"""
        logger.info("🚀 开始简化版批量OCR处理")
        logger.info("=" * 60)

        # 获取所有图片文件
        image_files = self.get_image_files()

        if not image_files:
            logger.error("没有找到可处理的图片文件")
            return

        # 处理每张图片
        all_results = []
        successful_count = 0

        for i, image_path in enumerate(image_files, 1):
            logger.info(f"\n📸 [{i}/{len(image_files)}] 处理图片")
            logger.info("-" * 40)

            try:
                result = self.process_single_image(image_path)
                all_results.append(result)

                # 检查是否有成功的识别结果
                has_success = any(
                    ocr_result.get('success', False)
                    for ocr_result in result['ocr_results'].values()
                )
                if has_success:
                    successful_count += 1

            except Exception as e:
                logger.error(f"处理图片失败 {os.path.basename(image_path)}: {e}")

        # 生成总结报告
        self.generate_summary_report(all_results, successful_count)

        logger.info("\n" + "=" * 60)
        logger.info("🎉 批量OCR处理完成！")
        logger.info(f"📊 处理统计: {successful_count}/{len(image_files)} 张图片识别成功")
        logger.info(f"📁 结果保存在: {self.output_folder}")

    def generate_summary_report(self, all_results, successful_count):
        """生成总结报告"""
        summary_file = self.output_folder / "batch_ocr_summary.json"

        summary_data = {
            'batch_info': {
                'total_images': len(all_results),
                'successful_images': successful_count,
                'processing_timestamp': datetime.now().isoformat(),
                'input_folder': str(self.input_folder),
                'output_folder': str(self.output_folder)
            },
            'engine_statistics': {},
            'detailed_results': all_results
        }

        # 统计各引擎成功率
        engine_stats = {}
        for result in all_results:
            for engine, ocr_result in result['ocr_results'].items():
                if engine not in engine_stats:
                    engine_stats[engine] = {'total': 0, 'successful': 0}
                engine_stats[engine]['total'] += 1
                if ocr_result.get('success', False):
                    engine_stats[engine]['successful'] += 1

        for engine, stats in engine_stats.items():
            success_rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
            summary_data['engine_statistics'][engine] = {
                'total_attempts': stats['total'],
                'successful_attempts': stats['successful'],
                'success_rate': f"{success_rate:.1f}%"
            }

        # 保存总结报告
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)

        # 生成文本版总结
        summary_txt = self.output_folder / "batch_ocr_summary.txt"
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write("批量OCR处理总结报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件夹: {self.input_folder}\n")
            f.write(f"输出文件夹: {self.output_folder}\n\n")

            f.write("处理统计:\n")
            f.write(f"  总图片数: {len(all_results)}\n")
            f.write(f"  成功识别: {successful_count}\n")
            f.write(f"  成功率: {(successful_count / len(all_results) * 100):.1f}%\n\n")

            f.write("引擎统计:\n")
            for engine, stats in summary_data['engine_statistics'].items():
                f.write(f"  {engine}:\n")
                f.write(f"    尝试次数: {stats['total_attempts']}\n")
                f.write(f"    成功次数: {stats['successful_attempts']}\n")
                f.write(f"    成功率: {stats['success_rate']}\n")

        logger.info(f"📋 总结报告已保存:")
        logger.info(f"    JSON版本: {summary_file.name}")
        logger.info(f"    文本版本: {summary_txt.name}")


def main():
    """主函数"""
    try:
        # 创建简化版批量处理器
        processor = SimpleBatchOCR(
            input_folder="pictures",
            output_folder="output_pictures"
        )

        # 开始批量处理
        processor.process_all_images()

    except KeyboardInterrupt:
        logger.info("\n⏹️  处理被用户中断")
    except Exception as e:
        logger.error(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
