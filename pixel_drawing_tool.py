#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
像素绘制工具
这是一个基于PyQt6的像素绘制工具，支持多种绘图功能，包括点、线、矩形、圆形、三角形、菱形、五边形、六边形、星形等基本图形的绘制，
以及图片导入功能。用户可以设置画布大小，生成对应的C语言代码。

依赖库：
- PyQt6: 用于图形界面
  安装命令: pip install PyQt6

作者: MistyRainDreamX
Github: https://github.com/2549850807/Pixel-Drawing
更新日期: 2025-10-31
"""

# 以下导入需要安装PyQt6库
# pip install PyQt6
import sys
import os
import math
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSpinBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QToolBar, QColorDialog, QSizePolicy, QScrollArea, QToolButton, QMenu, QFileDialog, QCheckBox, QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QRect, QPoint, QEvent
from PyQt6.QtGui import QPainter, QColor, QMouseEvent, QWheelEvent, QIcon, QPixmap, QAction, QActionGroup, QKeySequence, QShortcut


class PixelGridWidget(QWidget):
    TOOL_PENCIL = 1
    TOOL_LINE = 2
    TOOL_RECTANGLE = 3
    TOOL_CIRCLE = 4
    TOOL_TRIANGLE = 5
    TOOL_DIAMOND = 6
    TOOL_PENTAGON = 7
    TOOL_HEXAGON = 8
    TOOL_STAR = 9
    TOOL_IMAGE = 10
    
    def __init__(self, width=32, height=32, pixel_size=20):
        super().__init__()
        self.width = width
        self.height = height
        self.pixel_size = pixel_size
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.zoom_factor = 1.0
        self.current_tool = self.TOOL_PENCIL
        self.drawing_state = None
        self.start_pos = None
        self.temp_grid = None
        self.fill_modes = {
            self.TOOL_RECTANGLE: False,
            self.TOOL_CIRCLE: False,
            self.TOOL_TRIANGLE: False,
            self.TOOL_DIAMOND: False,
            self.TOOL_PENTAGON: False,
            self.TOOL_STAR: False,
            self.TOOL_HEXAGON: False
        }
        self.last_pos = None
        self.preview_end = None
        self.guideline_interval = 1
        
        self.image_data = None
        self.image_position = None
        self.image_scale = 1.0
        self.is_placing_image = False
        self.original_image_size = None
        
        self.history = []
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        
        self.update_grid_size()
        
    def set_grid_size(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.update_grid_size()
        
    def update_grid_size(self):
        target_width = int(self.width * self.pixel_size * self.zoom_factor)
        target_height = int(self.height * self.pixel_size * self.zoom_factor)
        
        if self.parent() and hasattr(self.parent(), 'viewport'):
            viewport = self.parent().viewport()
            viewport_width = viewport.width()
            viewport_height = viewport.height()
            
            min_zoom_x = viewport_width / (self.width * self.pixel_size)
            min_zoom_y = viewport_height / (self.height * self.pixel_size)
            min_zoom = max(min_zoom_x, min_zoom_y, 0.1)
            
            if self.zoom_factor < min_zoom:
                target_width = max(target_width, viewport_width)
                target_height = max(target_height, viewport_height)
        
        self.setFixedSize(target_width, target_height)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        actual_pixel_size = self.pixel_size * self.zoom_factor
        grid_to_draw = self.temp_grid if self.temp_grid is not None else self.grid

        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        painter.setPen(QColor(80, 80, 80))
        for x in range(0, self.width + 1, self.guideline_interval):
            x_pos = int(round(x * actual_pixel_size))
            painter.drawLine(x_pos, 0, x_pos, int(round(self.height * actual_pixel_size)))
        for y in range(0, self.height + 1, self.guideline_interval):
            y_pos = int(round(y * actual_pixel_size))
            painter.drawLine(0, y_pos, int(round(self.width * actual_pixel_size)), y_pos)

        w = self.width
        h = self.height
        ps = actual_pixel_size
        for y in range(h):
            row = grid_to_draw[y]
            x = 0
            while x < w:
                while x < w and not row[x]:
                    x += 1
                if x >= w:
                    break
                start = x
                while x < w and row[x]:
                    x += 1
                end = x - 1
                x0 = int(round(start * ps))
                x1 = int(round((end + 1) * ps))
                y0 = int(round(y * ps))
                y1 = int(round((y + 1) * ps))
                painter.fillRect(
                    QRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
                    Qt.GlobalColor.white
                )
        
        if self.current_tool == self.TOOL_IMAGE and self.is_placing_image and self.image_data and self.image_position:
            self.draw_image_preview(painter)
    
    def draw_image_preview(self, painter):
        if not self.image_data or not self.image_position:
            return
            
        start_x, start_y = self.image_position
        img_height = len(self.image_data)
        img_width = len(self.image_data[0]) if img_height > 0 else 0
        
        scaled_width = int(img_width * self.image_scale)
        scaled_height = int(img_height * self.image_scale)
        
        actual_pixel_size = self.pixel_size * self.zoom_factor
        
        max_preview_pixels = 20000
        total_pixels = scaled_width * scaled_height
        sample_factor = 1
        
        if total_pixels > max_preview_pixels:
            sample_factor = int(math.sqrt(total_pixels / max_preview_pixels))
            sample_factor = max(1, sample_factor)
        
        if img_width <= 200 and img_height <= 200:
            sample_factor = 1
        elif img_width <= 300 and img_height <= 300:
            sample_factor = max(1, sample_factor // 2)
        
        for y in range(0, scaled_height, sample_factor):
            for x in range(0, scaled_width, sample_factor):
                orig_x = int(x / self.image_scale)
                orig_y = int(y / self.image_scale)
                
                if (orig_x < img_width and orig_y < img_height and 
                    0 <= start_x + x < self.width and 0 <= start_y + y < self.height):
                    if self.image_data[orig_y][orig_x]:
                        x_pos = int(round((start_x + x) * actual_pixel_size))
                        y_pos = int(round((start_y + y) * actual_pixel_size))
                        rect_width = max(1, int(actual_pixel_size * sample_factor))
                        rect_height = max(1, int(actual_pixel_size * sample_factor))
                        
                        if rect_width > 4 and rect_height > 4:
                            painter.fillRect(
                                QRect(x_pos, y_pos, rect_width, rect_height),
                                QColor(200, 200, 200, 90)
                            )
                            painter.setPen(QColor(220, 220, 220, 130))
                            for gx in range(0, rect_width, max(2, rect_width // 3)):
                                painter.drawLine(x_pos + gx, y_pos, x_pos + gx, y_pos + rect_height)
                            for gy in range(0, rect_height, max(2, rect_height // 3)):
                                painter.drawLine(x_pos, y_pos + gy, x_pos + rect_width, y_pos + gy)
                        else:
                            painter.fillRect(
                                QRect(x_pos, y_pos, rect_width, rect_height),
                                QColor(200, 200, 200, 128)
                            )
        
        painter.setPen(QColor(100, 150, 255, 220))
        x_pos = int(round(start_x * actual_pixel_size))
        y_pos = int(round(start_y * actual_pixel_size))
        width = int(round(scaled_width * actual_pixel_size))
        height = int(round(scaled_height * actual_pixel_size))
        painter.drawRect(x_pos, y_pos, width, height)
        
        corner_size = min(12, max(4, width // 15, height // 15))
        painter.setBrush(QColor(100, 150, 255, 220))
        painter.drawRect(x_pos, y_pos, corner_size, corner_size)
    
    def _get_pixel_coordinates(self, pos):
        actual_pixel_size = self.pixel_size * self.zoom_factor
        
        x = int(pos.x() / actual_pixel_size)
        y = int(pos.y() / actual_pixel_size)
        
        return x, y
    
    def _draw_line(self, grid, start_pos, end_pos, state):
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = state
                
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
    def _draw_rectangle(self, grid, start_pos, end_pos, state, filled=False):
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)
        
        if filled:
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        grid[y][x] = state
        else:
            for x in range(min_x, max_x + 1):
                if 0 <= x < self.width:
                    if 0 <= min_y < self.height:
                        grid[min_y][x] = state
                    if 0 <= max_y < self.height:
                        grid[max_y][x] = state
                        
            for y in range(min_y + 1, max_y):
                if 0 <= y < self.height:
                    if 0 <= min_x < self.width:
                        grid[y][min_x] = state
                    if 0 <= max_x < self.width:
                        grid[y][max_x] = state
    
    def _draw_circle(self, grid, center, radius, state, filled=False):
        cx, cy = center
        if radius <= 0:
            return

        if filled:
            for y in range(-radius, radius + 1):
                rem = radius * radius - y * y
                if rem < 0:
                    continue
                x_max = int(round(math.sqrt(rem)))
                for x in range(-x_max, x_max + 1):
                    px, py = cx + x, cy + y
                    if 0 <= px < self.width and 0 <= py < self.height:
                        grid[py][px] = state

        steps = max(32, int(8 * radius))
        last = None
        for i in range(steps + 1):
            theta = 2.0 * math.pi * i / steps
            px = int(round(cx + radius * math.cos(theta)))
            py = int(round(cy + radius * math.sin(theta)))
            if 0 <= px < self.width and 0 <= py < self.height:
                if last is not None:
                    self._draw_line(grid, last, (px, py), state)
                else:
                    grid[py][px] = state
                last = (px, py)
    
    def _draw_triangle(self, grid, points, state, filled=False):
        if len(points) != 3:
            return
            
        if filled:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            min_y = max(0, min(y1, y2, y3))
            max_y = min(self.height - 1, max(y1, y2, y3))
            for y in range(min_y, max_y + 1):
                xs = []
                if (y1 != y2) and (min(y1, y2) <= y < max(y1, y2)):
                    xs.append(int(x1 + (y - y1) * (x2 - x1) / (y2 - y1)))
                if (y2 != y3) and (min(y2, y3) <= y < max(y2, y3)):
                    xs.append(int(x2 + (y - y2) * (x3 - x2) / (y3 - y2)))
                if (y3 != y1) and (min(y3, y1) <= y < max(y3, y1)):
                    xs.append(int(x3 + (y - y3) * (x1 - x3) / (y1 - y3)))
                if len(xs) >= 2:
                    xs.sort()
                    x_start = max(0, min(xs[0], xs[1]))
                    x_end = min(self.width - 1, max(xs[0], xs[1]))
                    for x in range(x_start, x_end + 1):
                        grid[y][x] = state

        else:
            self._draw_line(grid, points[0], points[1], state)
            self._draw_line(grid, points[1], points[2], state)
            self._draw_line(grid, points[2], points[0], state)
    
    def _draw_polygon(self, grid, points, state, filled=False):
        if not points or len(points) < 3:
            return
        if filled:
            min_y = max(0, min(p[1] for p in points))
            max_y = min(self.height - 1, max(p[1] for p in points))
            n = len(points)
            for y in range(min_y, max_y + 1):
                xs = []
                for i in range(n):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % n]
                    if y1 == y2:
                        continue
                    y_min = min(y1, y2)
                    y_max = max(y1, y2)
                    if y_min <= y < y_max:
                        x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        xs.append(x_intersect)
                xs.sort()
                for j in range(0, len(xs) - 1, 2):
                    left = xs[j]
                    right = xs[j + 1]
                    x_start = max(0, int(math.ceil(min(left, right))))
                    x_end = min(self.width - 1, int(math.floor(max(left, right))))
                    for x in range(x_start, x_end + 1):
                        grid[y][x] = state
            for vx, vy in points:
                if 0 <= vx < self.width and 0 <= vy < self.height:
                    grid[vy][vx] = state
        else:
            for i in range(len(points)):
                self._draw_line(grid, points[i], points[(i + 1) % len(points)], state)

    def _fill_flat_triangle(self, grid, p1, p2, p3, state):
        if p2[0] > p3[0]:
            p2, p3 = p3, p2
            
        if p2[1] != p1[1]:
            invslope1 = (p2[0] - p1[0]) / (p2[1] - p1[1])
        else:
            invslope1 = 0
            
        if p3[1] != p1[1]:
            invslope2 = (p3[0] - p1[0]) / (p3[1] - p1[1])
        else:
            invslope2 = 0
            
        curx1 = float(p1[0])
        curx2 = float(p1[0])
        
        for y in range(p1[1], p2[1] + 1):
            if 0 <= y < self.height:
                start_x = int(curx1)
                end_x = int(curx2)
                for x in range(start_x, end_x + 1):
                    if 0 <= x < self.width:
                        grid[y][x] = state
            curx1 += invslope1
            curx2 += invslope2
    
    def _clear_temp_grid(self):
        self.temp_grid = None
        self.start_pos = None
        
    def set_tool(self, tool):
        self.current_tool = tool
        self._clear_temp_grid()
        
        if tool != self.TOOL_IMAGE and self.is_placing_image:
            self.cancel_image_placement()
    
    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if not file_path:
            return False
            
        reply = QMessageBox.question(
            self, 
            "颜色反转", 
            "是否对导入的图片进行颜色反转？", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        invert_colors = (reply == QMessageBox.StandardButton.Yes)
            
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "错误", "无法加载图片")
            return False
            
        self.original_image_size = pixmap.size()
        
        max_width = min(self.width * 0.85, 500)
        max_height = min(self.height * 0.85, 500)
        
        pixmap = pixmap.scaled(
            int(max_width), int(max_height),
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        image = pixmap.toImage()
        
        if invert_colors:
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    inverted_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                    image.setPixelColor(x, y, inverted_color)
        
        width, height = image.width(), image.height()
        
        progress = QProgressDialog("正在处理图片...", "取消", 0, 5, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("图片处理进度")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        progress.setLabelText("正在分析图片亮度...")
        progress.setValue(1)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            return False
            
        brightness_map = []
        for y in range(height):
            row = []
            for x in range(width):
                color = image.pixelColor(x, y)
                brightness = (color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114)
                row.append(brightness)
            brightness_map.append(row)
        
        progress.setLabelText("正在增强图片对比度...")
        progress.setValue(2)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            return False
            
        enhanced_brightness_map = []
        
        scales = [1, 2, 3, 5, 7]
        scale_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        for y in range(height):
            enhanced_row = []
            for x in range(width):
                original_brightness = brightness_map[y][x]
                
                enhanced_value = 0
                total_weight = 0
                
                for i, scale in enumerate(scales):
                    local_sum = 0
                    local_count = 0
                    
                    for ky in range(max(0, y - scale), min(height, y + scale + 1)):
                        for kx in range(max(0, x - scale), min(width, x + scale + 1)):
                            local_sum += brightness_map[ky][kx]
                            local_count += 1
                    
                    if local_count > 0:
                        local_avg = local_sum / local_count
                        scale_enhanced = original_brightness + (original_brightness - local_avg) * 0.8
                        enhanced_value += scale_enhanced * scale_weights[i]
                        total_weight += scale_weights[i]
                
                if total_weight > 0:
                    enhanced_brightness = enhanced_value / total_weight
                else:
                    enhanced_brightness = original_brightness
                    
                enhanced_row.append(enhanced_brightness)
                
            enhanced_brightness_map.append(enhanced_row)
        
        progress.setLabelText("正在计算图片统计信息...")
        progress.setValue(3)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            return False
            
        total_brightness = 0
        brightness_values = []
        
        for y in range(height):
            for x in range(width):
                brightness = enhanced_brightness_map[y][x]
                total_brightness += brightness
                brightness_values.append(brightness)
                
        pixel_count = width * height
        avg_brightness = total_brightness / pixel_count if pixel_count > 0 else 128
        
        variance = 0
        for brightness in brightness_values:
            variance += (brightness - avg_brightness) ** 2
        std_dev = (variance / pixel_count) ** 0.5 if pixel_count > 0 else 0
        
        progress.setLabelText("正在转换为黑白图像...")
        progress.setValue(4)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            return False
            
        bw_data = [[False for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                enhanced_brightness = enhanced_brightness_map[y][x]
                
                local_threshold = avg_brightness
                
                if std_dev > 50:
                    local_threshold = avg_brightness * 0.6 + enhanced_brightness * 0.4
                elif std_dev > 35:
                    local_threshold = avg_brightness * 0.65 + enhanced_brightness * 0.35
                elif std_dev > 25:
                    local_threshold = avg_brightness * 0.7 + enhanced_brightness * 0.3
                elif std_dev > 15:
                    local_threshold = avg_brightness * 0.75 + enhanced_brightness * 0.25
                else:
                    local_threshold = avg_brightness * 0.65 + enhanced_brightness * 0.35
                
                edge_strength = 0
                if 2 < y < height - 3 and 2 < x < width - 3:
                    grad_x = (
                        (enhanced_brightness_map[y-1][x+1] - enhanced_brightness_map[y-1][x-1]) +
                        2 * (enhanced_brightness_map[y][x+1] - enhanced_brightness_map[y][x-1]) +
                        (enhanced_brightness_map[y+1][x+1] - enhanced_brightness_map[y+1][x-1])
                    )
                    grad_y = (
                        (enhanced_brightness_map[y+1][x-1] - enhanced_brightness_map[y-1][x-1]) +
                        2 * (enhanced_brightness_map[y+1][x] - enhanced_brightness_map[y-1][x]) +
                        (enhanced_brightness_map[y+1][x+1] - enhanced_brightness_map[y-1][x+1])
                    )
                    gradient = (grad_x ** 2 + grad_y ** 2) ** 0.5
                    edge_strength = gradient / 8
                    
                    if edge_strength > std_dev * 0.25:
                        local_threshold *= (0.8 + 0.2 * (min(edge_strength / (std_dev + 1), 1)))
                
                texture_factor = 1.0
                if 3 < y < height - 4 and 3 < x < width - 4:
                    local_variance = 0
                    local_mean = 0
                    count = 0
                    
                    for ky in range(y-3, y+4):
                        for kx in range(x-3, x+4):
                            local_mean += enhanced_brightness_map[ky][kx]
                            count += 1
                    
                    if count > 0:
                        local_mean /= count
                        
                        for ky in range(y-3, y+4):
                            for kx in range(x-3, x+4):
                                diff = enhanced_brightness_map[ky][kx] - local_mean
                                local_variance += diff * diff
                        
                        local_variance = (local_variance / count) ** 0.5
                        
                        if local_variance > std_dev * 0.3:
                            texture_factor = 0.85
                
                adjusted_threshold = local_threshold * texture_factor
                
                if enhanced_brightness > adjusted_threshold:
                    bw_data[y][x] = True
                else:
                    bw_data[y][x] = False
        
        progress.setLabelText("正在完成处理...")
        progress.setValue(5)
        QApplication.processEvents()
        
        self.image_data = bw_data
        
        self.is_placing_image = True
        self.image_scale = 1.0
        
        progress.close()
        
        return True
    
    def pixmap_to_bw_data_high_precision(self, pixmap):
        img = pixmap.toImage()
        width, height = img.width(), img.height()
        
        progress = QProgressDialog("正在处理图片数据...", "取消", 0, 4, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("图片处理进度")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        progress.setLabelText("正在分析图片亮度...")
        progress.setValue(1)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            progress.close()
            return [[False for _ in range(width)] for _ in range(height)]
            
        brightness_map = []
        for y in range(height):
            row = []
            for x in range(width):
                color = img.pixelColor(x, y)
                brightness = (color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114)
                row.append(brightness)
            brightness_map.append(row)
        
        progress.setLabelText("正在增强图片对比度...")
        progress.setValue(2)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            progress.close()
            return [[False for _ in range(width)] for _ in range(height)]
            
        enhanced_brightness_map = []
        
        scales = [1, 2, 3, 5, 7]
        scale_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        for y in range(height):
            enhanced_row = []
            for x in range(width):
                original_brightness = brightness_map[y][x]
                
                enhanced_value = 0
                total_weight = 0
                
                for i, scale in enumerate(scales):
                    local_sum = 0
                    local_count = 0
                    
                    for ky in range(max(0, y - scale), min(height, y + scale + 1)):
                        for kx in range(max(0, x - scale), min(width, x + scale + 1)):
                            local_sum += brightness_map[ky][kx]
                            local_count += 1
                    
                    if local_count > 0:
                        local_avg = local_sum / local_count
                        scale_enhanced = original_brightness + (original_brightness - local_avg) * 0.8
                        enhanced_value += scale_enhanced * scale_weights[i]
                        total_weight += scale_weights[i]
                
                if total_weight > 0:
                    enhanced_brightness = enhanced_value / total_weight
                else:
                    enhanced_brightness = original_brightness
                    
                enhanced_row.append(enhanced_brightness)
                
            enhanced_brightness_map.append(enhanced_row)
        
        progress.setLabelText("正在计算图片统计信息...")
        progress.setValue(3)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            progress.close()
            return [[False for _ in range(width)] for _ in range(height)]
            
        total_brightness = 0
        brightness_values = []
        
        for y in range(height):
            for x in range(width):
                brightness = enhanced_brightness_map[y][x]
                total_brightness += brightness
                brightness_values.append(brightness)
                
        pixel_count = width * height
        avg_brightness = total_brightness / pixel_count if pixel_count > 0 else 128
        
        variance = 0
        for brightness in brightness_values:
            variance += (brightness - avg_brightness) ** 2
        std_dev = (variance / pixel_count) ** 0.5 if pixel_count > 0 else 0
        
        progress.setLabelText("正在转换为黑白图像...")
        progress.setValue(4)
        QApplication.processEvents()
        
        if progress.wasCanceled():
            progress.close()
            return [[False for _ in range(width)] for _ in range(height)]
            
        bw_data = [[False for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                enhanced_brightness = enhanced_brightness_map[y][x]
                
                local_threshold = avg_brightness
                
                if std_dev > 50:
                    local_threshold = avg_brightness * 0.6 + enhanced_brightness * 0.4
                elif std_dev > 35:
                    local_threshold = avg_brightness * 0.65 + enhanced_brightness * 0.35
                elif std_dev > 25:
                    local_threshold = avg_brightness * 0.7 + enhanced_brightness * 0.3
                elif std_dev > 15:
                    local_threshold = avg_brightness * 0.75 + enhanced_brightness * 0.25
                else:
                    local_threshold = avg_brightness * 0.65 + enhanced_brightness * 0.35
                
                edge_strength = 0
                if 2 < y < height - 3 and 2 < x < width - 3:
                    grad_x = (
                        (enhanced_brightness_map[y-1][x+1] - enhanced_brightness_map[y-1][x-1]) +
                        2 * (enhanced_brightness_map[y][x+1] - enhanced_brightness_map[y][x-1]) +
                        (enhanced_brightness_map[y+1][x+1] - enhanced_brightness_map[y+1][x-1])
                    )
                    grad_y = (
                        (enhanced_brightness_map[y+1][x-1] - enhanced_brightness_map[y-1][x-1]) +
                        2 * (enhanced_brightness_map[y+1][x] - enhanced_brightness_map[y-1][x]) +
                        (enhanced_brightness_map[y+1][x+1] - enhanced_brightness_map[y-1][x+1])
                    )
                    gradient = (grad_x ** 2 + grad_y ** 2) ** 0.5
                    edge_strength = gradient / 8
                    
                    if edge_strength > std_dev * 0.25:
                        local_threshold *= (0.8 + 0.2 * (min(edge_strength / (std_dev + 1), 1)))
                
                texture_factor = 1.0
                if 3 < y < height - 4 and 3 < x < width - 4:
                    local_variance = 0
                    local_mean = 0
                    count = 0
                    
                    for ky in range(y-3, y+4):
                        for kx in range(x-3, x+4):
                            local_mean += enhanced_brightness_map[ky][kx]
                            count += 1
                    
                    if count > 0:
                        local_mean /= count
                        
                        for ky in range(y-3, y+4):
                            for kx in range(x-3, x+4):
                                diff = enhanced_brightness_map[ky][kx] - local_mean
                                local_variance += diff * diff
                        
                        local_variance = (local_variance / count) ** 0.5
                        
                        if local_variance > std_dev * 0.3:
                            texture_factor = 0.85
                
                adjusted_threshold = local_threshold * texture_factor
                
                if enhanced_brightness > adjusted_threshold:
                    bw_data[y][x] = True
                else:
                    bw_data[y][x] = False
                    
        progress.close()
                    
        return bw_data
    
    def pixmap_to_bw_data(self, pixmap):
        return self.pixmap_to_bw_data_high_precision(pixmap)
    
    def adjust_image_scale(self):
        if not self.image_data or not self.original_image_size:
            return
            
        img_width = len(self.image_data[0]) if self.image_data else 0
        img_height = len(self.image_data) if self.image_data else 0
        
        max_width = self.width * 0.8
        max_height = self.height * 0.8
        
        if img_width > max_width or img_height > max_height:
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            self.image_scale = min(scale_x, scale_y)
    
    def cancel_image_placement(self):
        self.is_placing_image = False
        self.image_data = None
        self.image_position = None
        self.image_scale = 1.0
        self.original_image_size = None
        self.update()
        
        mainWindow = self.window()
        if hasattr(mainWindow, 'switch_to_pencil_tool'):
            mainWindow.switch_to_pencil_tool()
    
    def place_image_on_grid(self):
        if not self.image_data or not self.image_position:
            return
            
        start_x, start_y = self.image_position
        img_height = len(self.image_data)
        img_width = len(self.image_data[0]) if img_height > 0 else 0
        
        scaled_width = int(img_width * self.image_scale)
        scaled_height = int(img_height * self.image_scale)
        
        for y in range(scaled_height):
            for x in range(scaled_width):
                orig_x = int(x / self.image_scale)
                orig_y = int(y / self.image_scale)
                
                if (orig_x < img_width and orig_y < img_height and 
                    0 <= start_x + x < self.width and 0 <= start_y + y < self.height):
                    self.grid[start_y + y][start_x + x] = self.image_data[orig_y][orig_x]
        
        mainWindow = self.window()
        if hasattr(mainWindow, 'switch_to_pencil_tool'):
            mainWindow.switch_to_pencil_tool()
    
    def set_fill_mode(self, tool, filled):
        if tool in self.fill_modes:
            self.fill_modes[tool] = filled
        
    def _snapshot(self):
        self.history.append([row[:] for row in self.grid])

    def _clamp_to_bounds(self, x, y):
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return x, y

    def mousePressEvent(self, event: QMouseEvent):
        if self.current_tool == self.TOOL_IMAGE and self.is_placing_image:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.image_data and self.image_position:
                    self._snapshot()
                    self.place_image_on_grid()
                    self.cancel_image_placement()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                self.cancel_image_placement()
                return
                
        x, y = self._get_pixel_coordinates(event.position())
        
        x, y = self._clamp_to_bounds(x, y)
            
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing_state = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.drawing_state = False
        else:
            return
            
        if self.current_tool == self.TOOL_PENCIL:
            self._snapshot()
            self.grid[y][x] = self.drawing_state
            self.last_pos = (x, y)
            self.update()
        else:
            self._snapshot()
            self.start_pos = (x, y)
            self.temp_grid = [row[:] for row in self.grid]
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.current_tool == self.TOOL_IMAGE and self.is_placing_image:
            if self.image_data:
                x, y = self._get_pixel_coordinates(event.position())
                
                if self.image_position is None or abs(self.image_position[0] - x) > 0 or abs(self.image_position[1] - y) > 0:
                    self.image_position = (x, y)
                    self.update()
            return
            
        x, y = self._get_pixel_coordinates(event.position())
        
        x, y = self._clamp_to_bounds(x, y)
            
        if self.current_tool == self.TOOL_PENCIL:
            if self.drawing_state is not None:
                if self.last_pos is not None:
                    self._draw_line(self.grid, self.last_pos, (x, y), self.drawing_state)
                else:
                    self.grid[y][x] = self.drawing_state
                self.last_pos = (x, y)
                self.update()
        elif self.start_pos is not None and self.temp_grid is not None:
            self.temp_grid = [row[:] for row in self.grid]
            
            if self.current_tool == self.TOOL_LINE:
                self._draw_line(self.temp_grid, self.start_pos, (x, y), self.drawing_state)
            elif self.current_tool == self.TOOL_RECTANGLE:
                self._draw_rectangle(self.temp_grid, self.start_pos, (x, y), self.drawing_state, self.fill_modes[self.TOOL_RECTANGLE])
            elif self.current_tool == self.TOOL_CIRCLE:
                dx = x - self.start_pos[0]
                dy = y - self.start_pos[1]
                radius = int(math.sqrt(dx*dx + dy*dy))
                self._draw_circle(self.temp_grid, self.start_pos, radius, self.drawing_state, self.fill_modes[self.TOOL_CIRCLE])
            elif self.current_tool == self.TOOL_TRIANGLE:
                points = [self.start_pos, (x, self.start_pos[1]), (self.start_pos[0], y)]
                self._draw_triangle(self.temp_grid, points, self.drawing_state, self.fill_modes[self.TOOL_TRIANGLE])
            elif self.current_tool == self.TOOL_DIAMOND:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                theta = math.atan2(dy, dx)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                base = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                pts = []
                for bx, by in base:
                    rx = int(round(cx + r * (bx * cos_t - by * sin_t)))
                    ry = int(round(cy + r * (bx * sin_t + by * cos_t)))
                    pts.append((rx, ry))
                self._draw_polygon(self.temp_grid, pts, self.drawing_state, self.fill_modes[self.TOOL_DIAMOND])
            elif self.current_tool == self.TOOL_PENTAGON:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                angle_offset = -math.pi / 2 + math.atan2(dy, dx)
                pts = []
                for i in range(5):
                    theta = angle_offset + 2 * math.pi * i / 5
                    px = int(round(cx + r * math.cos(theta)))
                    py = int(round(cy + r * math.sin(theta)))
                    pts.append((px, py))
                self._draw_polygon(self.temp_grid, pts, self.drawing_state, self.fill_modes[self.TOOL_PENTAGON])
            elif self.current_tool == self.TOOL_STAR:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r_outer = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                r_inner = max(1, int(round(r_outer * 0.381966)))
                angle_offset = -math.pi / 2 + math.atan2(dy, dx)
                pts = []
                for i in range(10):
                    r = r_outer if i % 2 == 0 else r_inner
                    theta = angle_offset + math.pi * i / 5
                    px = int(round(cx + r * math.cos(theta)))
                    py = int(round(cy + r * math.sin(theta)))
                    pts.append((px, py))
                self._draw_polygon(self.temp_grid, pts, self.drawing_state, self.fill_modes[self.TOOL_STAR])
            elif self.current_tool == self.TOOL_HEXAGON:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                angle_offset = -math.pi / 2 + math.atan2(dy, dx)
                pts = []
                for i in range(6):
                    theta = angle_offset + 2 * math.pi * i / 6
                    px = int(round(cx + r * math.cos(theta)))
                    py = int(round(cy + r * math.sin(theta)))
                    pts.append((px, py))
                self._draw_polygon(self.temp_grid, pts, self.drawing_state, self.fill_modes[self.TOOL_HEXAGON])

            self.preview_end = (x, y)
            self.update()
                
    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.current_tool == self.TOOL_IMAGE and self.is_placing_image:
            return
            
        if self.drawing_state is None:
            return
            
        x, y = self._get_pixel_coordinates(event.position())
        
        x, y = self._clamp_to_bounds(x, y)
            
        if self.current_tool == self.TOOL_PENCIL:
            self.grid[y][x] = self.drawing_state
            self.last_pos = None
        elif self.start_pos is not None:
            if self.current_tool == self.TOOL_LINE:
                self._draw_line(self.grid, self.start_pos, (x, y), self.drawing_state)
            elif self.current_tool == self.TOOL_RECTANGLE:
                self._draw_rectangle(self.grid, self.start_pos, (x, y), self.drawing_state, self.fill_modes[self.TOOL_RECTANGLE])
            elif self.current_tool == self.TOOL_CIRCLE:
                dx = x - self.start_pos[0]
                dy = y - self.start_pos[1]
                radius = int(math.sqrt(dx*dx + dy*dy))
                self._draw_circle(self.grid, self.start_pos, radius, self.drawing_state, self.fill_modes[self.TOOL_CIRCLE])
            elif self.current_tool == self.TOOL_TRIANGLE:
                points = [self.start_pos, (x, self.start_pos[1]), (self.start_pos[0], y)]
                self._draw_triangle(self.grid, points, self.drawing_state, self.fill_modes[self.TOOL_TRIANGLE])
            elif self.current_tool == self.TOOL_DIAMOND:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                theta = math.atan2(dy, dx)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                base = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                pts = []
                for bx, by in base:
                    rx = int(round(cx + r * (bx * cos_t - by * sin_t)))
                    ry = int(round(cy + r * (bx * sin_t + by * cos_t)))
                    pts.append((rx, ry))
                self._draw_polygon(self.grid, pts, self.drawing_state, self.fill_modes[self.TOOL_DIAMOND])
            elif self.current_tool == self.TOOL_PENTAGON:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                angle_offset = -math.pi / 2 + math.atan2(dy, dx)
                pts = []
                for i in range(5):
                    theta = angle_offset + 2 * math.pi * i / 5
                    px = int(round(cx + r * math.cos(theta)))
                    py = int(round(cy + r * math.sin(theta)))
                    pts.append((px, py))
                self._draw_polygon(self.grid, pts, self.drawing_state, self.fill_modes[self.TOOL_PENTAGON])
            elif self.current_tool == self.TOOL_STAR:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r_outer = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                r_inner = max(1, int(round(r_outer * 0.381966)))
                angle_offset = -math.pi / 2 + math.atan2(dy, dx)
                pts = []
                for i in range(10):
                    r = r_outer if i % 2 == 0 else r_inner
                    theta = angle_offset + math.pi * i / 5
                    px = int(round(cx + r * math.cos(theta)))
                    py = int(round(cy + r * math.sin(theta)))
                    pts.append((px, py))
                self._draw_polygon(self.grid, pts, self.drawing_state, self.fill_modes[self.TOOL_STAR])
            elif self.current_tool == self.TOOL_HEXAGON:
                cx, cy = self.start_pos
                dx, dy = x - cx, y - cy
                r = max(1, int(round(math.sqrt(dx*dx + dy*dy))))
                angle_offset = -math.pi / 2 + math.atan2(dy, dx)
                pts = []
                for i in range(6):
                    theta = angle_offset + 2 * math.pi * i / 6
                    px = int(round(cx + r * math.cos(theta)))
                    py = int(round(cy + r * math.sin(theta)))
                    pts.append((px, py))
                self._draw_polygon(self.grid, pts, self.drawing_state, self.fill_modes[self.TOOL_HEXAGON])

        self._clear_temp_grid()
        self.preview_end = None
        self.drawing_state = None
        self.update()
            
    def wheelEvent(self, event: QWheelEvent):
        if (self.current_tool == self.TOOL_IMAGE and self.is_placing_image and 
            self.image_data and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier)):
            delta = event.angleDelta().y()
            
            if delta > 0:
                self.image_scale *= 1.1
            elif delta < 0:
                self.image_scale /= 1.1
                
            self.image_scale = max(0.1, min(self.image_scale, 5.0))
            
            self.update()
            return
            
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            
            if delta > 0:
                self.zoom_factor *= 1.1
            elif delta < 0:
                self.zoom_factor /= 1.1
                
            if self.parent() and hasattr(self.parent(), 'viewport'):
                viewport = self.parent().viewport()
                viewport_width = viewport.width()
                viewport_height = viewport.height()
                
                canvas_width = self.width * self.pixel_size * self.zoom_factor
                canvas_height = self.height * self.pixel_size * self.zoom_factor
                
                min_zoom_x = viewport_width / (self.width * self.pixel_size)
                min_zoom_y = viewport_height / (self.height * self.pixel_size)
                min_zoom = max(min_zoom_x, min_zoom_y, 0.05)
                
                max_zoom = max(20.0, min_zoom * 50)
                
                self.zoom_factor = max(min_zoom, min(self.zoom_factor, max_zoom))
            else:
                self.zoom_factor = max(0.05, min(self.zoom_factor, 20.0))
            
            self.update_grid_size()
            self.update()
        else:
            super().wheelEvent(event)
            
    def clear_grid(self):
        if hasattr(self, 'history'):
            self.history.append([row[:] for row in self.grid])
        self.grid = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.update()

    def set_guideline_interval(self, interval):
        self.guideline_interval = interval
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("像素绘制工具")
        self.setGeometry(100, 100, 800, 600)
        
        icon_path = os.path.join(os.path.dirname(__file__), "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        QApplication.instance().installEventFilter(self)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.create_toolbar()
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.pixel_grid = PixelGridWidget(32, 32, 20)
        self.scroll_area.setWidget(self.pixel_grid)
        main_layout.addWidget(self.scroll_area)
        
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        width_label = QLabel("宽度：")
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 999)
        self.width_spinbox.setValue(32)
        self.width_spinbox.valueChanged.connect(self.update_grid_size)
        self.width_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        
        height_label = QLabel("高度：")
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 999)
        self.height_spinbox.setValue(32)
        self.height_spinbox.valueChanged.connect(self.update_grid_size)
        self.height_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        
        guideline_label = QLabel("辅助线间隔：")
        self.guideline_spinbox = QSpinBox()
        self.guideline_spinbox.setRange(1, 100)
        self.guideline_spinbox.setValue(1)
        self.guideline_spinbox.valueChanged.connect(self.update_guideline_interval)
        self.guideline_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        
        clear_button = QPushButton("清屏")
        clear_button.clicked.connect(self.clear_grid)

        create_button = QPushButton("生成代码")
        create_button.clicked.connect(self.create_pixel_files)
        
        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        delete_shortcut.activated.connect(self.clear_grid)
        
        control_layout.addWidget(width_label)
        control_layout.addWidget(self.width_spinbox)
        control_layout.addWidget(height_label)
        control_layout.addWidget(self.height_spinbox)
        control_layout.addWidget(guideline_label)
        control_layout.addWidget(self.guideline_spinbox)
        control_layout.addWidget(clear_button)
        control_layout.addWidget(create_button)
        control_layout.addStretch()
        
        main_layout.addWidget(control_panel)

        undo_shortcut = QShortcut(QKeySequence('Ctrl+Z'), self)
        undo_shortcut.activated.connect(self.undo_last_action)
        
    def undo_last_action(self):
        if hasattr(self.pixel_grid, 'history') and self.pixel_grid.history:
            self.pixel_grid.grid = self.pixel_grid.history.pop()
            self.pixel_grid.update()
        
    def create_pixel_files(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not directory:
            return
        
        lines = []
        lines.append('void Pixel_Image_Draw(void);\n\n')
        lines.append('#define Pixel_Draw(x, y) my_pixel_draw(x, y)\n\n')
        lines.append('void Pixel_Image_Draw(void) {\n')
        lines.append('    // 根据画布（以左上角为原点）逐像素输出 Pixel_Draw(x, y);\n')
        grid = self.pixel_grid.grid
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        for y in range(height):
            for x in range(width):
                if grid[y][x]:
                    lines.append(f"    Pixel_Draw({x}, {y});\n")
        lines.append('}\n')
        c_content = ''.join(lines)
        
        c_path = os.path.join(directory, 'my_pixel.c')
        with open(c_path, 'w', encoding='utf-8', newline='') as f:
            f.write(c_content)
        
        try:
            import subprocess
            subprocess.Popen(['notepad.exe', c_path])
        except Exception:
            pass
        
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and self.isActiveWindow():
            if isinstance(event, QWheelEvent) and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.pixel_grid.wheelEvent(event)
                return True
        return super().eventFilter(obj, event)
        
    def create_toolbar(self):
        toolbar = QToolBar("Tools")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        
        pencil_action = QAction("点", self)
        pencil_action.setCheckable(True)
        pencil_action.setChecked(True)
        pencil_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_PENCIL))
        tool_group.addAction(pencil_action)
        toolbar.addAction(pencil_action)
        
        line_action = QAction("直线", self)
        line_action.setCheckable(True)
        line_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_LINE))
        tool_group.addAction(line_action)
        toolbar.addAction(line_action)
        
        rect_action = QAction("矩形", self)
        rect_action.setCheckable(True)
        rect_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_RECTANGLE))
        tool_group.addAction(rect_action)
        toolbar.addAction(rect_action)
        
        circle_action = QAction("圆形", self)
        circle_action.setCheckable(True)
        circle_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_CIRCLE))
        tool_group.addAction(circle_action)
        toolbar.addAction(circle_action)
        
        triangle_action = QAction("三角形", self)
        triangle_action.setCheckable(True)
        triangle_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_TRIANGLE))
        tool_group.addAction(triangle_action)
        toolbar.addAction(triangle_action)

        diamond_action = QAction("菱形", self)
        diamond_action.setCheckable(True)
        diamond_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_DIAMOND))
        tool_group.addAction(diamond_action)
        toolbar.addAction(diamond_action)

        pentagon_action = QAction("五边形", self)
        pentagon_action.setCheckable(True)
        pentagon_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_PENTAGON))
        tool_group.addAction(pentagon_action)
        toolbar.addAction(pentagon_action)

        hexagon_action = QAction("六边形", self)
        hexagon_action.setCheckable(True)
        hexagon_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_HEXAGON))
        tool_group.addAction(hexagon_action)
        toolbar.addAction(hexagon_action)

        star_action = QAction("星形", self)
        star_action.setCheckable(True)
        star_action.triggered.connect(lambda: self.set_tool(PixelGridWidget.TOOL_STAR))
        tool_group.addAction(star_action)
        toolbar.addAction(star_action)
        
        image_action = QAction("图片", self)
        image_action.setCheckable(True)
        image_action.triggered.connect(self.import_image)
        toolbar.addAction(image_action)
        
        toolbar.addSeparator()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar.addWidget(spacer)
        
        self.fill_checkbox = QCheckBox("填充")
        self.fill_checkbox.setChecked(False)
        self.fill_checkbox.toggled.connect(self.on_fill_toggled)
        toolbar.addWidget(self.fill_checkbox)
        
        self.pencil_action = pencil_action
        self.line_action = line_action
        self.rect_action = rect_action
        self.circle_action = circle_action
        self.triangle_action = triangle_action
        self.diamond_action = diamond_action
        self.pentagon_action = pentagon_action
        self.star_action = star_action
        self.hexagon_action = hexagon_action
        self.image_action = image_action
        
        self.shape_tool_actions = [self.rect_action, self.circle_action, self.triangle_action,
                                   self.diamond_action, self.pentagon_action, self.star_action, self.hexagon_action]
        
    def set_tool(self, tool):
        self.pixel_grid.set_tool(tool)
        
    def import_image(self):
        self.pixel_grid.set_tool(PixelGridWidget.TOOL_IMAGE)
        self.image_action.setChecked(True)
        
        success = self.pixel_grid.import_image()
        if not success:
            self.switch_to_pencil_tool()
    
    def switch_to_pencil_tool(self):
        self.pixel_grid.set_tool(PixelGridWidget.TOOL_PENCIL)
        self.pencil_action.setChecked(True)
        self.image_action.setChecked(False)
    
    def on_fill_toggled(self, checked):
        for tool in [PixelGridWidget.TOOL_RECTANGLE, PixelGridWidget.TOOL_CIRCLE, 
                     PixelGridWidget.TOOL_TRIANGLE, PixelGridWidget.TOOL_DIAMOND,
                     PixelGridWidget.TOOL_PENTAGON, PixelGridWidget.TOOL_STAR,
                     PixelGridWidget.TOOL_HEXAGON]:
            self.pixel_grid.set_fill_mode(tool, checked)
            
        self.pixel_grid.set_fill_mode(self.pixel_grid.current_tool, checked)
        
    def update_grid_size(self):
        width = self.width_spinbox.value()
        height = self.height_spinbox.value()
        self.pixel_grid.set_grid_size(width, height)
        
    def update_guideline_interval(self):
        interval = self.guideline_spinbox.value()
        self.pixel_grid.set_guideline_interval(interval)
        
    def clear_grid(self):
        self.pixel_grid.clear_grid()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()