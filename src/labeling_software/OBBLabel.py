import os
import sys # Needed to get script directory reliably
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import colorsys # For generating distinct colors
import copy # For deep copying points during drag
import platform
IS_MAC = platform.system() == "Darwin"

# --- Data Structures ---

class BoundingBox:
    """Represents a single bounding box with a class ID and four corner points."""
    def __init__(self, class_id, points):
        self.class_id = int(class_id)
        self.points = [(float(p[0]), float(p[1])) for p in points] # Image coordinates

    def get_normalized_corners(self, img_width, img_height):
        """Returns corners as normalized coordinates (0-1 range)."""
        if not img_width or not img_height:
            return [(0, 0)] * 4
        return [(p[0] / img_width, p[1] / img_height) for p in self.points]

    def get_center(self):
        """Calculates the geometric center of the bounding box."""
        if not self.points: return 0, 0
        num_points = len(self.points)
        if num_points == 0: return 0, 0
        center_x = sum(p[0] for p in self.points) / num_points
        center_y = sum(p[1] for p in self.points) / num_points
        return center_x, center_y

    def calculate_area(self):
        """Calculates the area using the Shoelace formula."""
        n = len(self.points)
        if n < 3: return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0

# --- Utility Functions ---

def hsv_to_rgb_hex(h, s, v):
    """Converts HSV color to Tkinter hex color string."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def point_in_polygon(x, y, poly_points):
    """Checks if a point (x, y) is inside a polygon using the Ray Casting algorithm."""
    n = len(poly_points)
    if n < 3: return False
    inside = False
    p1x, p1y = poly_points[0]
    for i in range(n + 1):
        p2x, p2y = poly_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y: # Edge is not horizontal
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else: # Edge is horizontal
                        xinters = p1x # Value doesn't strictly matter here if y == p1y
                    # Check vertical edges and points to the left
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# --- Core Classes ---

class ClassManager:
    """Manages object classes, loading/saving them from/to 'classes.txt' in the script's directory."""
    def __init__(self):
        # --- FIX: Determine classes.txt path based on script location ---
        if getattr(sys, 'frozen', False):
            # If running as a bundled executable (e.g., PyInstaller)
            script_dir = os.path.dirname(sys.executable)
        else:
            # If running as a normal script (__file__ should be defined)
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError: # Fallback if __file__ is not defined (e.g., interactive)
                script_dir = os.getcwd()
        self.classes_file = os.path.join(script_dir, "classes.txt")
        # --- End FIX ---
        self.classes = {}  # {id: name}
        self.colors = {}   # {id: hex_color}
        self.load_classes() # Load classes immediately on initialization

    # --- REMOVED: set_directory method is no longer needed for classes.txt ---

    def load_classes(self):
        """Loads classes from the classes.txt file in the script directory."""
        self.classes = {}
        self.colors = {}
        # Check if the determined classes_file path exists
        if self.classes_file and os.path.exists(self.classes_file):
            try:
                # Open with UTF-8 encoding for broader compatibility
                with open(self.classes_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Ignore empty lines and lines starting with '#' (comments)
                        if line and not line.startswith('#'):
                            parts = line.split(' ', 1) # Split only on the first space
                            # Ensure we have exactly two parts and the first is a digit
                            if len(parts) == 2 and parts[0].isdigit():
                                class_id = int(parts[0])
                                # Optionally skip class ID 0 if it causes issues
                                if class_id == 0: continue
                                class_name = parts[1]
                                self.classes[class_id] = class_name
                                self._assign_color(class_id) # Assign a color
            except Exception as e:
                # Show error message if loading fails
                messagebox.showerror("Error Loading Classes", f"Could not load classes from {self.classes_file}:\n{e}")
        # else:
            # Optionally print a message if the file doesn't exist on first run
            # print(f"Note: Classes file not found at {self.classes_file}. It will be created on first save.")

    def save_classes(self):
        """Saves the current classes to the classes.txt file in the script directory."""
        if not self.classes_file:
            # This should not happen with the __init__ fix, but handle defensively
            print("Error: classes_file path not set. Cannot save classes.")
            return
        try:
            # Ensure the directory exists (it should, but doesn't hurt to check)
            os.makedirs(os.path.dirname(self.classes_file), exist_ok=True)
            # Write classes to file, sorted by ID
            with open(self.classes_file, 'w', encoding='utf-8') as f:
                for class_id, class_name in sorted(self.classes.items()):
                    f.write(f"{class_id} {class_name}\n")
        except Exception as e:
            messagebox.showerror("Error Saving Classes", f"Could not save classes to {self.classes_file}:\n{e}")

    def add_class(self, class_name):
        """Adds a new class, assigning the next available ID."""
        # Check for empty name or duplicates (case-insensitive)
        if not class_name or class_name.lower() in [name.lower() for name in self.classes.values()]:
            return None
        # Find the next available ID (starting from 1)
        next_id = 1
        while next_id in self.classes:
            next_id += 1
        # Add class, assign color, and save
        self.classes[next_id] = class_name
        self._assign_color(next_id)
        self.save_classes()
        return next_id

    def remove_class(self, class_id):
        """Removes a class by its ID."""
        if class_id in self.classes:
            del self.classes[class_id]
            # Also remove associated color if it exists
            if class_id in self.colors: del self.colors[class_id]
            self.save_classes() # Save changes to file
            return True
        return False # Class ID not found

    def get_class_name(self, class_id):
        """Gets the class name for a given ID."""
        return self.classes.get(class_id, f"Unknown ({class_id})")

    def get_class_id(self, class_name):
        """Gets the class ID for a given name (case-insensitive)."""
        search_name = class_name.lower()
        for class_id, name in self.classes.items():
            if name.lower() == search_name: return class_id
        return None # Not found

    def get_all_classes(self):
        """Returns a dictionary of all classes {id: name}."""
        return self.classes.copy()

    def get_class_color(self, class_id):
        """Gets the display color for a class ID."""
        # Assign color if missing (e.g., loaded labels with unknown class)
        if class_id not in self.colors: self._assign_color(class_id)
        # Return assigned color or default grey if still missing (e.g., ID 0)
        return self.colors.get(class_id, "#808080")

    def _assign_color(self, class_id):
        """Assigns a visually distinct color to a class ID."""
        if class_id == 0: return # Don't assign color to ID 0
        # Use golden angle approximation for distinct hues
        hue = (class_id * 0.61803398875) % 1.0
        # Use fixed saturation and value for bright colors
        self.colors[class_id] = hsv_to_rgb_hex(hue, 0.85, 0.9)


class ImageCanvas(tk.Canvas):
    """Canvas for displaying images and drawing/editing bounding boxes."""
    HANDLE_SIZE = 5 # Pixel radius for handles
    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0

    def __init__(self, parent, class_manager, status_updater, **kwargs):
        super().__init__(parent, bg="gray20", highlightthickness=0, **kwargs)
        self.parent = parent
        self.class_manager = class_manager
        self.status_updater = status_updater

        # Image state
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.image_id = None
        self.img_width, self.img_height = 0, 0
        self.scale_factor = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0

        # Bounding box state
        self.bounding_boxes = []
        self.current_class_id = None
        self.changes_made = False # Flag for tracking modifications

        # Drawing state
        self.drawing_polygon_points = []

        # Selection and Editing state
        self.selected_bbox_index = None
        self.selected_point_index = None
        self.is_dragging_selection = False
        self.drag_start_canvas_x = 0
        self.drag_start_canvas_y = 0
        self.drag_start_points = []

        # Panning state
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._is_panning = False

        self._setup_bindings()

    def _setup_bindings(self):
        self.bind("<Configure>", self._on_resize)

        # --- Drawing / Dragging ---
        self.bind("<ButtonPress-1>",  self._on_left_press)
        self.bind("<B1-Motion>",      self._on_left_drag)
        self.bind("<ButtonRelease-1>",self._on_left_release)

        # --- Selection ---
        self.bind("<Command-ButtonPress-1>", self._on_select_press)  # ⌘‑click (mac)
        self.bind("<Control-ButtonPress-1>", self._on_select_press)  # Ctrl‑click (non‑mac)
        if IS_MAC:
            self.bind("<ButtonPress-3>", self._on_select_press)      # ⌃‑click becomes Button‑3

        # --- Panning ---
        self.bind("<ButtonPress-2>",  self._start_pan)               # physical middle button
        self.bind("<B2-Motion>",      self._do_pan)
        self.bind("<ButtonRelease-2>",self._end_pan)
        # Alt + Left works on all OSes (Option on mac)
        self.bind("<Alt-ButtonPress-1>",     self._start_pan)
        self.bind("<Alt-B1-Motion>",         self._do_pan)
        self.bind("<Alt-ButtonRelease-1>",   self._end_pan)

        self.bind("<Motion>", self._on_mouse_move)


    def load_image(self, image_path):
        """Loads an image, fits it, and loads labels."""
        try:
            if image_path is None:
                self.pil_image, self.image_path = None, None
                self._reset_state()
                self.update_display()
                return True

            self.pil_image = Image.open(image_path).convert("RGBA")
            self.img_width, self.img_height = self.pil_image.size
            self.image_path = image_path
            self._reset_state()
            self.load_labels()
            self.changes_made = False # Reset changes flag
            self.after_idle(self.fit_image_to_window)
            self.status_updater(f"Loaded: {os.path.basename(image_path)}")
            return True
        except FileNotFoundError:
            messagebox.showerror("Error", f"Image file not found:\n{image_path}")
            return False
        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Could not load image:\n{image_path}\n{e}")
            self.pil_image, self.image_path = None, None
            self._reset_state()
            self.update_display()
            return False

    def fit_image_to_window(self):
        """Calculates scale factor to fit the image within the canvas."""
        if not self.pil_image: return
        canvas_width, canvas_height = self.winfo_width(), self.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
             self.after(50, self.fit_image_to_window); return
        if self.img_width == 0 or self.img_height == 0: return

        pad_w, pad_h = max(1, canvas_width - 20), max(1, canvas_height - 20)
        scale = min(pad_w / self.img_width, pad_h / self.img_height, 1.0) # Don't exceed 1.0 initially
        self.scale_factor = max(self.MIN_ZOOM, min(self.MAX_ZOOM, scale))
        self.update_display()

    def update_display(self, zoom_center_img=None, zoom_center_canvas=None):
        """Redraws the image and bounding boxes."""
        view_x, view_y = self.xview(), self.yview() # Store current view
        self.delete("all"); self.image_id, self.tk_image = None, None # Clear

        if not self.pil_image:
            self.configure(scrollregion=(0, 0, self.winfo_width(), self.winfo_height())); return

        cw, ch = self.winfo_width(), self.winfo_height()
        sw, sh = int(self.img_width * self.scale_factor), int(self.img_height * self.scale_factor)
        if sw <= 0 or sh <= 0: return

        self.image_offset_x = max(0, (cw - sw) // 2)
        self.image_offset_y = max(0, (ch - sh) // 2)

        try: # Resize and create PhotoImage
            resized = self.pil_image.resize((sw, sh), Image.Resampling.NEAREST)
            self.tk_image = ImageTk.PhotoImage(resized)
        except Exception as e: print(f"Error resizing/creating image: {e}"); self.tk_image = None

        # --- Zoom centering ---
        target_vx, target_vy = view_x[0], view_y[0] # Default to old view
        if zoom_center_img and zoom_center_canvas and self.tk_image:
            img_x, img_y = zoom_center_img
            canv_x, canv_y = zoom_center_canvas # Scrolled coords where mouse was
            new_cx_unsc = img_x * self.scale_factor + self.image_offset_x
            new_cy_unsc = img_y * self.scale_factor + self.image_offset_y
            delta_x, delta_y = new_cx_unsc - canv_x, new_cy_unsc - canv_y
            total_sw = max(cw, sw + self.image_offset_x)
            total_sh = max(ch, sh + self.image_offset_y)
            curr_vx_px = view_x[0] * total_sw
            curr_vy_px = view_y[0] * total_sh
            target_vx_px, target_vy_px = curr_vx_px + delta_x, curr_vy_px + delta_y
            if total_sw > cw: target_vx = max(0.0, min(target_vx_px / total_sw, (total_sw - cw) / total_sw))
            else: target_vx = 0.0
            if total_sh > ch: target_vy = max(0.0, min(target_vy_px / total_sh, (total_sh - ch) / total_sh))
            else: target_vy = 0.0
        # --- End Zoom centering ---

        if self.tk_image: # Draw image
            self.image_id = self.create_image(self.image_offset_x, self.image_offset_y, anchor=tk.NW, image=self.tk_image, tags="image")

        # Update scroll region
        scroll_w, scroll_h = max(cw, sw + self.image_offset_x), max(ch, sh + self.image_offset_y)
        self.configure(scrollregion=(0, 0, scroll_w, scroll_h))

        # Restore/apply scroll position
        self.xview_moveto(target_vx); self.yview_moveto(target_vy)

        # Draw annotations
        self._draw_all_bounding_boxes()
        self._draw_temp_polygon_points()
        self._draw_zoom_indicator()

    def set_zoom(self, scale_delta, event=None):
        """Applies zoom, optionally centering on the mouse event."""
        if not self.pil_image: return
        old_scale = self.scale_factor
        new_scale = max(self.MIN_ZOOM, min(self.MAX_ZOOM, old_scale + scale_delta))
        if abs(new_scale - old_scale) < 0.001: return

        center_img, center_canv = None, None
        if event: # Get coords for centering if event provided
            canv_x, canv_y = self.canvasx(event.x), self.canvasy(event.y)
            center_canv = (canv_x, canv_y)
            center_img = self.canvas_to_image_coords(*center_canv)
            if not center_img: center_canv = None # Don't center if outside image

        self.scale_factor = new_scale
        self.update_display(center_img, center_canv) # Redraw with centering info
        self.status_updater(f"Zoom: {self.scale_factor:.1f}x")

    # --- Coordinate Conversion ---

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Converts scrolled canvas coordinates to image coordinates."""
        if not self.pil_image or self.scale_factor == 0: return None
        img_x = (canvas_x - self.image_offset_x) / self.scale_factor
        img_y = (canvas_y - self.image_offset_y) / self.scale_factor
        if 0 <= img_x <= self.img_width and 0 <= img_y <= self.img_height:
            return max(0, min(self.img_width, img_x)), max(0, min(self.img_height, img_y))
        return None

    def image_to_canvas_coords(self, img_x, img_y):
        """Converts image coordinates to unscrolled canvas coordinates."""
        if not self.pil_image: return None
        return img_x * self.scale_factor + self.image_offset_x, img_y * self.scale_factor + self.image_offset_y

    # --- Drawing Helpers ---

    def _draw_all_bounding_boxes(self):
        self.delete("bbox"); # Clear existing boxes/handles/labels
        for i, bbox in enumerate(self.bounding_boxes): self._draw_single_bounding_box(i, bbox)

    def _draw_single_bounding_box(self, index, bbox):
        is_selected = (index == self.selected_bbox_index)
        color = self.class_manager.get_class_color(bbox.class_id)
        line_width = 3 if is_selected else 2
        canvas_points = [self.image_to_canvas_coords(p[0], p[1]) for p in bbox.points]
        if None in canvas_points: return

        if len(canvas_points) >= 3: # Draw polygon
            flat = [c for p in canvas_points for c in p]
            self.create_polygon(*flat, outline=color, fill="", width=line_width, tags=("bbox", f"bbox_{index}"))
        elif len(canvas_points) == 2: # Draw line
             self.create_line(*canvas_points[0], *canvas_points[1], fill=color, width=line_width, tags=("bbox", f"bbox_{index}"))

        if len(canvas_points) == 4: # Draw label and handles if complete
            center_x, center_y = self.image_to_canvas_coords(*bbox.get_center())
            if center_x is not None:
                name = self.class_manager.get_class_name(bbox.class_id)
                fsize = 9; tw = len(name) * (fsize * 0.6)
                bg_id = self.create_rectangle(center_x - tw/2 - 2, center_y - fsize*0.8, center_x + tw/2 + 2, center_y + fsize*0.8, fill="black", outline="", tags=("bbox", f"bbox_{index}_label_bg"))
                lbl_id = self.create_text(center_x, center_y, text=name, fill=color, font=("Arial", fsize, "bold"), anchor=tk.CENTER, tags=("bbox", f"bbox_{index}_label"))
                self.tag_raise(lbl_id, bg_id)
            if is_selected: # Draw handles
                for j, (px, py) in enumerate(canvas_points):
                    hcol = "red" if j == self.selected_point_index else "white"
                    x1, y1 = px - self.HANDLE_SIZE, py - self.HANDLE_SIZE
                    x2, y2 = px + self.HANDLE_SIZE, py + self.HANDLE_SIZE
                    self.create_oval(x1, y1, x2, y2, fill=hcol, outline=color, width=1, tags=("bbox", f"bbox_{index}_handle", f"handle_{j}"))

    def _draw_temp_polygon_points(self):
        self.delete("temp_point", "temp_line")
        if not self.drawing_polygon_points or self.current_class_id is None: return
        color = self.class_manager.get_class_color(self.current_class_id)
        canvas_points = [self.image_to_canvas_coords(p[0], p[1]) for p in self.drawing_polygon_points]
        if None in canvas_points: return
        for i, (px, py) in enumerate(canvas_points): # Draw points
            self.create_oval(px-3, py-3, px+3, py+3, fill=color, outline=color, tags="temp_point")
        if len(canvas_points) > 1: # Draw lines
            for i in range(len(canvas_points) - 1):
                p1, p2 = canvas_points[i], canvas_points[i+1]
                self.create_line(*p1, *p2, fill=color, width=2, tags="temp_line")

    def _update_preview_lines(self, event_canvas_x, event_canvas_y):
        self.delete("preview_line")
        if not self.drawing_polygon_points or self.current_class_id is None or self.is_dragging_selection: return
        npts = len(self.drawing_polygon_points)
        if not (1 <= npts <= 3): return
        color = self.class_manager.get_class_color(self.current_class_id)
        last_cp = self.image_to_canvas_coords(*self.drawing_polygon_points[-1])
        if not last_cp: return
        self.create_line(*last_cp, event_canvas_x, event_canvas_y, fill=color, dash=(5, 3), width=2, tags="preview_line")
        if npts == 3:
            first_cp = self.image_to_canvas_coords(*self.drawing_polygon_points[0])
            if not first_cp: return
            self.create_line(event_canvas_x, event_canvas_y, *first_cp, fill=color, dash=(5, 3), width=2, tags="preview_line")

    def _draw_zoom_indicator(self):
        self.delete("zoom_indicator")
        txt = f"{self.scale_factor:.1f}x"
        vx, vy = self.canvasx(0), self.canvasy(0)
        shadow = self.create_text(vx + 11, vy + 11, text=txt, fill="black", anchor=tk.NW, font=("Arial", 10), tags="zoom_indicator")
        text = self.create_text(vx + 10, vy + 10, text=txt, fill="white", anchor=tk.NW, font=("Arial", 10), tags="zoom_indicator")
        self.tag_raise(text, shadow)

    # --- Bounding Box Creation ---

    def set_current_class(self, class_id):
        self.current_class_id = class_id
        if self.drawing_polygon_points: self.cancel_drawing(); self.status_updater("Drawing cancelled.")

    def start_drawing_polygon(self, img_coords):
        if self.current_class_id is None:
            self.status_updater("Select class first.", error=True); self._flash_warning("Select class!"); return
        self.cancel_selection(); self.drawing_polygon_points = [img_coords]; self._draw_temp_polygon_points()

    def add_polygon_point(self, img_coords):
        if not self.drawing_polygon_points: return
        self.drawing_polygon_points.append(img_coords); self._draw_temp_polygon_points()
        if len(self.drawing_polygon_points) == 4: self.finalize_polygon()

    def finalize_polygon(self):
        if len(self.drawing_polygon_points) == 4 and self.current_class_id is not None:
            self.bounding_boxes.append(BoundingBox(self.current_class_id, list(self.drawing_polygon_points)))
            self.changes_made = True
            self.status_updater(f"Added: {self.class_manager.get_class_name(self.current_class_id)}")
        self.cancel_drawing(redraw_boxes=True)

    def cancel_drawing(self, redraw_boxes=False):
        self.drawing_polygon_points = []; self.delete("temp_point", "temp_line", "preview_line")
        if redraw_boxes: self._draw_all_bounding_boxes()

    # --- Bounding Box Selection and Editing ---

    def find_item_at(self, canvas_x, canvas_y):
        """Finds topmost handle or box under scrolled canvas coords."""
        # First check if we're near any corners of a selected bounding box
        if self.selected_bbox_index is not None:
            bbox = self.bounding_boxes[self.selected_bbox_index]
            for corner_idx, point in enumerate(bbox.points):
                # Convert corner point to canvas coordinates
                canvas_point = self.image_to_canvas_coords(point[0], point[1])
                if canvas_point:
                    cx, cy = canvas_point
                    # Calculate distance from mouse to corner
                    distance = ((canvas_x - cx)**2 + (canvas_y - cy)**2)**0.5
                    # If within 20px, select this corner
                    if distance <= 20:
                        return "handle", self.selected_bbox_index, corner_idx
        
        # Check for existing handle objects in the canvas
        radius = self.HANDLE_SIZE + 2
        overlapping = self.find_overlapping(canvas_x - radius, canvas_y - radius, 
                                        canvas_x + radius, canvas_y + radius)
        
        # Check handles first
        for item_id in reversed(overlapping):
            tags = self.gettags(item_id)
            if "handle" in tags:
                p_idx, b_idx = -1, -1
                for tag in tags:
                    if tag.startswith("handle_"):
                        try: p_idx = int(tag.split("_")[1])
                        except: continue
                    elif tag.startswith("bbox_"):
                        try: b_idx = int(tag.split("_")[1])
                        except: continue
                if p_idx != -1 and b_idx != -1 and 0 <= b_idx < len(self.bounding_boxes):
                    coords = self.coords(item_id)
                    if coords:
                        cx, cy = (coords[0]+coords[2])/2, (coords[1]+coords[3])/2
                        if abs(canvas_x - cx) <= radius and abs(canvas_y - cy) <= radius:
                            return "handle", b_idx, p_idx
        
        # Check if we're within any box (for all boxes, not just selected)
        img_coords = self.canvas_to_image_coords(canvas_x, canvas_y)
        if img_coords:
            candidates = []
            for i, bbox in reversed(list(enumerate(self.bounding_boxes))):
                # First check corners of every box (not just selected)
                for corner_idx, point in enumerate(bbox.points):
                    canvas_point = self.image_to_canvas_coords(point[0], point[1])
                    if canvas_point:
                        cx, cy = canvas_point
                        distance = ((canvas_x - cx)**2 + (canvas_y - cy)**2)**0.5
                        if distance <= 20:
                            return "handle", i, corner_idx
                            
                # If no corners matched, check if inside the box
                if point_in_polygon(img_coords[0], img_coords[1], bbox.points):
                    candidates.append({'index': i, 'area': bbox.calculate_area()})
            
            if candidates:
                candidates.sort(key=lambda b: b['area']) # Smallest area first
                return "box", candidates[0]['index'], None
        
        return None, None, None # Nothing found

    def select_box_or_handle(self, canvas_x, canvas_y):
        """Selects item under cursor. Returns True if a selectable item was found."""
        item_type, b_idx, p_idx = self.find_item_at(canvas_x, canvas_y)
        needs_redraw = False
        if item_type == "handle":
            if self.selected_bbox_index != b_idx or self.selected_point_index != p_idx:
                self.selected_bbox_index, self.selected_point_index = b_idx, p_idx
                needs_redraw = True; self.status_updater(f"Selected handle {p_idx} of box {b_idx}")
        elif item_type == "box":
            if self.selected_bbox_index != b_idx or self.selected_point_index is not None:
                self.selected_bbox_index, self.selected_point_index = b_idx, None
                needs_redraw = True; self.status_updater(f"Selected box {b_idx} ({self.class_manager.get_class_name(self.bounding_boxes[b_idx].class_id)})")
        else: # Clicked empty space
            if self.selected_bbox_index is not None:
                self.cancel_selection(); needs_redraw = True # cancel_selection handles redraw internally now
        if needs_redraw: self._draw_all_bounding_boxes() # Redraw if selection changed
        return item_type is not None # Return True if handle or box was under cursor

    def start_dragging_selection(self, canvas_x, canvas_y):
        """Initiates dragging for the selected item."""
        if self.selected_bbox_index is None or not (0 <= self.selected_bbox_index < len(self.bounding_boxes)): return False
        if self.selected_point_index is None or (0 <= self.selected_point_index < 4): # Valid selection
            self.is_dragging_selection = True
            self.drag_start_canvas_x, self.drag_start_canvas_y = canvas_x, canvas_y
            self.drag_start_points = copy.deepcopy(self.bounding_boxes[self.selected_bbox_index].points)
            return True
        self.cancel_selection(); return False # Invalid point index

    def drag_selection_update(self, canvas_x, canvas_y):
        """Updates the position of the dragged item."""
        if not self.is_dragging_selection or self.selected_bbox_index is None or not self.drag_start_points: return
        if not (0 <= self.selected_bbox_index < len(self.bounding_boxes)): self.cancel_selection(); return

        delta_cx, delta_cy = canvas_x - self.drag_start_canvas_x, canvas_y - self.drag_start_canvas_y
        if self.scale_factor == 0: return
        delta_ix, delta_iy = delta_cx / self.scale_factor, delta_cy / self.scale_factor
        bbox = self.bounding_boxes[self.selected_bbox_index]

        if self.selected_point_index is not None: # Dragging handle
             if not (0 <= self.selected_point_index < 4): self.cancel_selection(); return
             orig_p = self.drag_start_points[self.selected_point_index]
             nx = max(0, min(self.img_width, orig_p[0] + delta_ix))
             ny = max(0, min(self.img_height, orig_p[1] + delta_iy))
             bbox.points[self.selected_point_index] = (nx, ny)
        else: # Dragging box
            new_pts = []
            for p in self.drag_start_points:
                nx = max(0, min(self.img_width, p[0] + delta_ix))
                ny = max(0, min(self.img_height, p[1] + delta_iy))
                new_pts.append((nx, ny))
            bbox.points = new_pts
        self._draw_all_bounding_boxes() # Update visuals

    def end_dragging_selection(self):
        """Finalizes the drag operation."""
        if self.is_dragging_selection:
             if self.selected_bbox_index is not None and 0 <= self.selected_bbox_index < len(self.bounding_boxes):
                 # Check if points actually changed
                 if self.bounding_boxes[self.selected_bbox_index].points != self.drag_start_points:
                     self.changes_made = True # Mark changes if modification occurred
             self.is_dragging_selection = False; self.drag_start_points = []

    def cancel_selection(self):
        """Deselects any selected item."""
        needs_redraw = self.selected_bbox_index is not None
        self.selected_bbox_index, self.selected_point_index = None, None
        if self.is_dragging_selection: self.is_dragging_selection = False; self.drag_start_points = []
        if needs_redraw: self._draw_all_bounding_boxes()

    def delete_selected_bbox(self):
        """Deletes the currently selected bounding box."""
        if self.selected_bbox_index is not None and 0 <= self.selected_bbox_index < len(self.bounding_boxes):
            try:
                name = self.class_manager.get_class_name(self.bounding_boxes[self.selected_bbox_index].class_id)
                idx = self.selected_bbox_index
                del self.bounding_boxes[self.selected_bbox_index]
                self.changes_made = True
                self.status_updater(f"Deleted box {idx} ({name})")
                self.cancel_selection(); return True # cancel_selection redraws
            except IndexError: self.status_updater("Error deleting box.", error=True); self.cancel_selection(); return False
        return False # Nothing selected

    # --- Label Saving/Loading ---

    def get_label_path(self):
        if not self.image_path: return None
        img_dir, img_name = os.path.dirname(self.image_path), os.path.basename(self.image_path)
        name_no_ext = os.path.splitext(img_name)[0]
        labels_dir = os.path.join(img_dir, "labels") # Labels in subdir
        return os.path.join(labels_dir, f"{name_no_ext}.txt")

    def save_labels(self):
        """Saves bounding boxes. Returns True on success/user-cancel, False on error."""
        label_file = self.get_label_path()
        if not label_file: messagebox.showwarning("Save Error", "Cannot get label path."); return False
        if not self.bounding_boxes:
             if messagebox.askyesno("Save Empty?", f"No boxes drawn for {os.path.basename(self.image_path)}.\nSave empty label file?"):
                 try:
                     os.makedirs(os.path.dirname(label_file), exist_ok=True)
                     open(label_file, 'w').close() # Create empty file
                     self.changes_made = False; self.status_updater(f"Saved empty: {os.path.basename(label_file)}"); return True
                 except Exception as e: messagebox.showerror("Save Error", f"Could not save empty file:\n{e}"); return False
             else: self.status_updater("Save cancelled."); return True # User cancel is not error
        try: # Save actual boxes
            os.makedirs(os.path.dirname(label_file), exist_ok=True)
            with open(label_file, 'w', encoding='utf-8') as f:
                if not self.img_width or not self.img_height: messagebox.showerror("Save Error", "Image dims zero."); return False
                for bbox in self.bounding_boxes:
                    norm = bbox.get_normalized_corners(self.img_width, self.img_height)
                    line = [str(bbox.class_id)] + [f"{c:.6f}" for p in norm for c in p]
                    f.write(" ".join(line) + "\n")
            self.changes_made = False; self.status_updater(f"Labels saved: {os.path.basename(label_file)}"); return True
        except Exception as e: messagebox.showerror("Save Error", f"Could not save labels:\n{e}"); return False

    def load_labels(self):
        label_file = self.get_label_path(); self.bounding_boxes = []
        if not label_file or not os.path.exists(label_file): return
        try:
            if not self.img_width or not self.img_height: print("W: Cannot load labels, img dims zero."); return
            unknown_ids = set(); lines_read = 0
            with open(label_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split()
                    if len(parts) == 9:
                        try:
                            cid = int(parts[0]); norm = [float(p) for p in parts[1:]]
                            if not all(0.0 <= c <= 1.0 for c in norm): print(f"W: L{i} Invalid coords: {line}"); continue
                            pts = [(norm[j]*self.img_width, norm[j+1]*self.img_height) for j in range(0, 8, 2)]
                            if cid not in self.class_manager.classes: unknown_ids.add(cid)
                            self.bounding_boxes.append(BoundingBox(cid, pts)); lines_read += 1
                        except ValueError: print(f"W: L{i} Invalid number: {line}")
                    else: print(f"W: L{i} Invalid parts: {line}")
            if lines_read > 0:
                 msg = f"Loaded {lines_read} boxes"
                 if unknown_ids: msg += f" (Unknown IDs: {sorted(list(unknown_ids))})"
                 self.status_updater(msg)
        except Exception as e: messagebox.showerror("Load Error", f"Could not load labels:\n{e}"); self.bounding_boxes = []

    # --- Internal Event Handlers ---

    def _on_resize(self, event=None): self.after(20, lambda: self.update_display())

    def _on_left_press(self, event):
        cx, cy = self.canvasx(event.x), self.canvasy(event.y)
        img_coords = self.canvas_to_image_coords(cx, cy)
        if self._is_panning: return
        if img_coords: # Click inside image
            if not self.drawing_polygon_points: self.start_drawing_polygon(img_coords)
            else: self.add_polygon_point(img_coords)
        else: # Click outside image
            if self.drawing_polygon_points: self.cancel_drawing(True); self.status_updater("Drawing cancelled.")
            elif self.selected_bbox_index is not None: self.cancel_selection()

    def _on_select_press(self, event):
         cx, cy = self.canvasx(event.x), self.canvasy(event.y)
         if self._is_panning: return
         if self.drawing_polygon_points: self.cancel_drawing(True)
         # Attempt selection and start drag if successful
         if self.select_box_or_handle(cx, cy):
             self.start_dragging_selection(cx, cy)

    def _on_left_drag(self, event):
        cx, cy = self.canvasx(event.x), self.canvasy(event.y)
        if self.is_dragging_selection: # Dragging selected item
            self.drag_selection_update(cx, cy)
        elif self.drawing_polygon_points: # Drawing new polygon
             img_coords = self.canvas_to_image_coords(cx, cy)
             if img_coords: self._update_preview_lines(cx, cy)
             else: self.delete("preview_line")

    def _on_mouse_move(self, event): # Update preview only when drawing and not dragging
        if self.drawing_polygon_points and not self.is_dragging_selection:
             cx, cy = self.canvasx(event.x), self.canvasy(event.y)
             img_coords = self.canvas_to_image_coords(cx, cy)
             if img_coords: self._update_preview_lines(cx, cy)
             else: self.delete("preview_line")

    def _on_left_release(self, event): # Finalize drag
        if self.is_dragging_selection: self.end_dragging_selection()

    def _start_pan(self, event):
        if not self.pil_image or self.drawing_polygon_points or self.is_dragging_selection: return
        self._is_panning = True; self._pan_start_x, self._pan_start_y = event.x, event.y; self.config(cursor="fleur")

    def _do_pan(self, event):
        if not self._is_panning: return
        dx, dy = event.x - self._pan_start_x, event.y - self._pan_start_y
        self.xview_scroll(-dx, "units"); self.yview_scroll(-dy, "units")
        self._pan_start_x, self._pan_start_y = event.x, event.y
        self._draw_zoom_indicator() # Keep zoom indicator visible

    def _end_pan(self, event):
        if self._is_panning: self._is_panning = False; self.config(cursor="")

    # --- Utility ---

    def _reset_state(self):
        self.cancel_drawing(); self.cancel_selection(); self.bounding_boxes = []
        self.scale_factor = 1.0; self.image_offset_x = 0; self.image_offset_y = 0
        self.changes_made = False; self.xview_moveto(0); self.yview_moveto(0)

    def _flash_warning(self, text):
        if not self.winfo_exists(): return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1: return
        vx, vy = self.canvasx(0), self.canvasy(0)
        x, y = vx + cw / 2, vy + 30
        bg = self.create_rectangle(x-100, y-15, x+100, y+15, fill="yellow", outline="black", tags="warning")
        txt = self.create_text(x, y, text=text, fill="black", font=("Arial", 12, "bold"), anchor=tk.CENTER, tags="warning")
        self.tag_raise(txt, bg); self.after(1500, lambda: self.delete("warning"))


class ClassDialog(tk.Toplevel):
    """Modal dialog for adding a new class name."""
    def __init__(self, parent, existing_class_names):
        super().__init__(parent)
        self.title("Add New Class"); self.resizable(False, False)
        self.transient(parent); self.grab_set(); self.result = None
        self.existing_lower = [name.lower() for name in existing_class_names]
        frame = ttk.Frame(self, padding="15"); frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Enter New Class Name:").pack(anchor=tk.W)
        self.entry = ttk.Entry(frame, width=35); self.entry.pack(fill=tk.X, pady=5); self.entry.focus_set()
        self.err_lbl = ttk.Label(frame, text="", foreground="red", justify=tk.LEFT, wraplength=260); self.err_lbl.pack(anchor=tk.W, pady=(0, 10), fill=tk.X)
        btn_frame = ttk.Frame(frame); btn_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM); btn_frame.columnconfigure(0, weight=1)
        ttk.Button(btn_frame, text="OK", command=self._ok, style="Accent.TButton", width=8).grid(row=0, column=1, padx=(0,5))
        ttk.Button(btn_frame, text="Cancel", command=self._cancel, width=8).grid(row=0, column=2)
        self.bind("<Return>", lambda e: self._ok()); self.bind("<Escape>", lambda e: self._cancel())
        self.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty(); pw, ph = parent.winfo_width(), parent.winfo_height()
        dw, dh = self.winfo_width(), self.winfo_height(); x, y = px + (pw - dw)//2, py + (ph - dh)//2
        self.geometry(f"+{x}+{y}"); self.wait_window(self)

    def _validate(self):
        name = self.entry.get().strip(); err = ""
        if not name: err = "Name cannot be empty."
        elif name.lower() in self.existing_lower: err = "Name already exists."
        elif name.isdigit(): err = "Name cannot be purely numeric."
        elif any(c in name for c in '<>:"/\\|?* '): err = f"Invalid chars found."
        self.err_lbl.config(text=err); return not err

    def _ok(self):
        if self._validate(): self.result = self.entry.get().strip(); self.destroy()
        else: self.entry.focus_set(); self.bell()

    def _cancel(self): self.result = None; self.destroy()


class MainApplication(tk.Tk):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.title("Bounding Box Labeler"); self.geometry("1200x800"); self.minsize(800, 600)
        self._setup_style()
        self.class_manager = ClassManager() # Manages classes.txt in script dir
        self.image_dir, self.image_files, self.current_image_index = None, [], -1
        self._resize_timer, self._status_clear_timer = None, None
        self._setup_ui()
        self._setup_bindings()
        self._update_ui_state(); self._update_classes_listbox() # Load classes into UI
        self.status_update("Select image folder to begin.", timeout=0)

    def _setup_style(self):
        self.style = ttk.Style()
        try: theme = "clam"; self.style.theme_use(theme)
        except tk.TclError: theme = self.style.theme_use() # Get default if clam fails
        print(f"Using theme: {theme}")
        try: # Define Accent button style
            self.style.configure("Accent.TButton", foreground="white", background="#0078D7")
            self.style.map("Accent.TButton", background=[('active', '#005A9E')], foreground=[('active', 'white')])
        except tk.TclError: print("W: Could not configure Accent.TButton style.")

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        # Left Panel
        left = ttk.Frame(self, width=250, padding=10); left.grid(row=0, column=0, sticky="nsew"); left.grid_propagate(False); left.grid_rowconfigure(3, weight=1)
        ttk.Button(left, text="📂 Select Folder...", command=self._select_folder, style="Accent.TButton").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        info_nav = ttk.Frame(left); info_nav.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5); info_nav.columnconfigure(0, weight=1)
        self.lbl_info = ttk.Label(info_nav, text="No folder", justify=tk.CENTER, anchor=tk.CENTER, relief=tk.GROOVE, padding=5, wraplength=220); self.lbl_info.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        nav_btns = ttk.Frame(info_nav); nav_btns.grid(row=1, column=0, columnspan=2, sticky="ew"); nav_btns.columnconfigure((0,1), weight=1)
        self.btn_prev = ttk.Button(nav_btns, text="⬅️ Prev", command=self._prev_image, width=8); self.btn_prev.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.btn_next = ttk.Button(nav_btns, text="Next ➡️", command=self._next_image, width=8); self.btn_next.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        ttk.Label(left, text="Classes:").grid(row=2, column=0, columnspan=2, sticky="w", pady=(15, 2))
        cls_frame = ttk.Frame(left); cls_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 5)); cls_frame.rowconfigure(0, weight=1); cls_frame.columnconfigure(0, weight=1)
        self.lst_cls = tk.Listbox(cls_frame, exportselection=False, activestyle="dotbox"); self.lst_cls.grid(row=0, column=0, sticky="nsew")
        cls_scroll = ttk.Scrollbar(cls_frame, orient=tk.VERTICAL, command=self.lst_cls.yview); cls_scroll.grid(row=0, column=1, sticky="ns"); self.lst_cls.config(yscrollcommand=cls_scroll.set); self.lst_cls.bind("<<ListboxSelect>>", self._on_class_select)
        cls_btns = ttk.Frame(left); cls_btns.grid(row=4, column=0, columnspan=2, sticky="ew"); cls_btns.columnconfigure((0,1), weight=1)
        ttk.Button(cls_btns, text="➕ Add", command=self._add_class, width=8).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.btn_rem = ttk.Button(cls_btns, text="➖ Remove", command=self._remove_class, width=8); self.btn_rem.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.btn_save = ttk.Button(left, text="💾 Save Labels", command=self._save_labels); self.btn_save.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(15, 0))
        # Canvas Area
        canv_frame = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=1); canv_frame.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=(10,0)); canv_frame.grid_rowconfigure(0, weight=1); canv_frame.grid_columnconfigure(0, weight=1)
        vscroll = ttk.Scrollbar(canv_frame, orient=tk.VERTICAL); vscroll.grid(row=0, column=1, sticky="ns"); hscroll = ttk.Scrollbar(canv_frame, orient=tk.HORIZONTAL); hscroll.grid(row=1, column=0, sticky="ew")
        self.canvas = ImageCanvas(canv_frame, self.class_manager, self.status_update, xscrollcommand=hscroll.set, yscrollcommand=vscroll.set); self.canvas.grid(row=0, column=0, sticky="nsew"); vscroll.config(command=self.canvas.yview); hscroll.config(command=self.canvas.xview)
        # Status Bar
        self.lbl_status = ttk.Label(self, text="", relief=tk.SUNKEN, anchor=tk.W, padding=(5, 3)); self.lbl_status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5,10))

    def _setup_bindings(self):
        self.bind("<Configure>", self._on_window_resize)
        # Use bind_all for global shortcuts
        self.bind_all("<Escape>", lambda e: self.canvas.cancel_drawing() if hasattr(self, 'canvas') else None)
        self.bind_all("<Delete>", lambda e: self.canvas.delete_selected_bbox() if hasattr(self, 'canvas') else None)
        self.bind_all("<BackSpace>", lambda e: self.canvas.delete_selected_bbox() if hasattr(self, 'canvas') else None)
        self.bind_all("<Left>", lambda e: self._prev_image())
        self.bind_all("<Right>", lambda e: self._next_image())
        # Platform-specific save shortcut
        save_mod = "Command" if self.tk.call('tk', 'windowingsystem') == "aqua" else "Control"
        self.bind_all(f"<{save_mod}-s>", lambda e: self._save_labels())
        self.bind_all(f"<{save_mod}-S>", lambda e: self._save_labels()) # Allow Shift modifier
        # Zoom bindings on canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel) # Windows/macOS Wheel
        self.canvas.bind("<Button-4>", self._on_mousewheel)   # Linux Scroll Up
        self.canvas.bind("<Button-5>", self._on_mousewheel)   # Linux Scroll Down
        self.canvas.bind("<Control-MouseWheel>", self._on_mousewheel) # Some trackpads/systems
        self.canvas.bind("<Command-MouseWheel>", self._on_mousewheel) # macOS trackpad pinch/zoom

    # --- Core Logic ---

    def _select_folder(self):
        if hasattr(self, 'canvas') and self.canvas.image_path and self.canvas.changes_made:
             if not self._prompt_and_save(): return # Cancelled
        folder = filedialog.askdirectory(title="Select Image Folder", initialdir=self.image_dir)
        if not folder: return # User cancelled
        self.image_dir = folder; self.status_update(f"Selected: {os.path.basename(folder)}")
        # Class list updated once in __init__ now
        self.image_files = []
        valid_ext = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        try:
            entries = sorted([e for e in os.scandir(self.image_dir) if e.is_file() and os.path.splitext(e.name)[1].lower() in valid_ext], key=lambda e: e.name)
            self.image_files = [e.path for e in entries]
        except OSError as e: messagebox.showerror("Error", f"Read folder error:\n{e}"); self.image_dir, self.image_files = None, []
        if self.image_files: self.current_image_index = 0; self._load_current_image()
        else: self.current_image_index = -1; self.canvas.load_image(None); messagebox.showwarning("No Images", "No supported images found.")
        self._update_ui_state()

    def _load_current_image(self):
        if not hasattr(self, 'canvas'): return
        path = self.image_files[self.current_image_index] if 0 <= self.current_image_index < len(self.image_files) else None
        if path and not self.canvas.load_image(path): # Try loading
            messagebox.showerror("Load Error", f"Failed load: {os.path.basename(path)}. Skipping.")
            idx = self.current_image_index
            try: self.image_files.pop(idx)
            except IndexError: pass
            if not self.image_files: self.current_image_index = -1; self.canvas.load_image(None)
            elif idx >= len(self.image_files): self.current_image_index = len(self.image_files) - 1
            self._load_current_image(); return # Recurse
        elif not path: self.canvas.load_image(None) # Clear if no path
        self._update_image_info(); self._update_ui_state()

    def _save_labels(self):
        if hasattr(self, 'canvas') and self.current_image_index != -1: return self.canvas.save_labels()
        else: self.status_update("No image loaded.", error=True); return False

    def _prompt_and_save(self):
        if hasattr(self, 'canvas') and self.canvas.image_path and self.canvas.changes_made:
            name = os.path.basename(self.canvas.image_path)
            resp = messagebox.askyesnocancel("Unsaved Changes", f"Save changes to '{name}'?", parent=self)
            if resp is True: return self._save_labels() # Proceed only if save OK
            elif resp is False: return True # Proceed without saving
            else: self.status_update("Cancelled."); return False # Cancel action
        return True # No changes, proceed

    def _next_image(self):
        if not self.image_files or self.current_image_index == -1: return
        if self.current_image_index >= len(self.image_files) - 1: self.status_update("Last image."); return
        if not self._prompt_and_save(): return
        self.current_image_index += 1; self._load_current_image()

    def _prev_image(self):
        if not self.image_files or self.current_image_index == -1: return
        if self.current_image_index <= 0: self.status_update("First image."); return
        if not self._prompt_and_save(): return
        self.current_image_index -= 1; self._load_current_image()

    def _update_classes_listbox(self):
        if not hasattr(self, 'lst_cls'): return
        self.lst_cls.delete(0, tk.END)
        sel_id = self.canvas.current_class_id if hasattr(self.canvas, 'current_class_id') else None
        sel_idx = -1; idx = 0
        for cid, name in sorted(self.class_manager.get_all_classes().items()):
            color = self.class_manager.get_class_color(cid)
            self.lst_cls.insert(tk.END, f"{cid}: {name}")
            try: # Set item colors
                 rgb = [int(c, 16)/255. for c in (color[1:3], color[3:5], color[5:7])]
                 lum = colorsys.rgb_to_hls(*rgb)[1]
                 fg = 'white' if lum < 0.5 else 'black'
                 self.lst_cls.itemconfig(idx, {'bg': color, 'fg': fg, 'selectbackground': color, 'selectforeground': fg})
            except Exception as e: print(f"W: Listbox color fail {idx} ({cid}): {e}")
            if cid == sel_id: sel_idx = idx
            idx += 1
        if sel_idx != -1: self.lst_cls.selection_set(sel_idx); self.lst_cls.activate(sel_idx); self.lst_cls.see(sel_idx)
        self._update_ui_state()

    def _on_class_select(self, event=None):
        if not hasattr(self, 'lst_cls') or not hasattr(self, 'canvas'): return
        sel = self.lst_cls.curselection()
        if sel:
            try: cid = int(self.lst_cls.get(sel[0]).split(":")[0]); self.canvas.set_current_class(cid); self.status_update(f"Selected: {self.class_manager.get_class_name(cid)}")
            except Exception as e: self.canvas.set_current_class(None); self.status_update("Error selecting class.", error=True); print(f"Class select error: {e}")
        else: self.canvas.set_current_class(None); self.status_update("No class selected.")
        self._update_ui_state()

    def _add_class(self):
        dialog = ClassDialog(self, self.class_manager.get_all_classes().values())
        if dialog.result:
            new_id = self.class_manager.add_class(dialog.result)
            if new_id:
                self.status_update(f"Added: {dialog.result} (ID: {new_id})"); self._update_classes_listbox()
                for i in range(self.lst_cls.size()): # Auto-select new class
                     if self.lst_cls.get(i).startswith(f"{new_id}: "): self.lst_cls.selection_clear(0, tk.END); self.lst_cls.selection_set(i); self.lst_cls.activate(i); self.lst_cls.see(i); self._on_class_select(); break

    def _remove_class(self):
        if not hasattr(self, 'lst_cls') or not hasattr(self, 'canvas'): return
        sel = self.lst_cls.curselection()
        if not sel: self.status_update("Select class to remove.", error=True); return
        try: cid = int(self.lst_cls.get(sel[0]).split(":")[0]); name = self.class_manager.get_class_name(cid)
        except: self.status_update("Invalid selection.", error=True); return
        used = any(b.class_id == cid for b in self.canvas.bounding_boxes)
        msg = f"Remove class '{name}' (ID: {cid})?" + ("\n\nWarning: Used in current image!" if used else "")
        if messagebox.askyesno("Confirm Removal", msg, parent=self):
            if self.class_manager.remove_class(cid):
                self.status_update(f"Removed: {name}")
                if self.canvas.current_class_id == cid: self.canvas.set_current_class(None)
                self._update_classes_listbox(); self.canvas.update_display()
            else: self.status_update(f"Failed remove: {name}.", error=True)

    def _update_ui_state(self):
        c = hasattr(self, 'canvas'); l = hasattr(self, 'lst_cls')
        img = c and self.canvas.image_path is not None
        prev = img and self.current_image_index > 0
        nxt = img and self.current_image_index < len(self.image_files) - 1
        cls = l and bool(self.lst_cls.curselection())
        if hasattr(self, 'btn_prev'): self.btn_prev.config(state=tk.NORMAL if prev else tk.DISABLED)
        if hasattr(self, 'btn_next'): self.btn_next.config(state=tk.NORMAL if nxt else tk.DISABLED)
        if hasattr(self, 'btn_rem'): self.btn_rem.config(state=tk.NORMAL if cls else tk.DISABLED)
        if hasattr(self, 'btn_save'): self.btn_save.config(state=tk.NORMAL if img else tk.DISABLED)

    def _update_image_info(self):
        if not hasattr(self, 'lbl_info'): return
        info = "No folder selected."
        if self.image_dir: info = "Folder selected."
        if self.image_files: info = "No images found."
        if self.current_image_index != -1: name = os.path.basename(self.image_files[self.current_image_index]); info = f"Image {self.current_image_index + 1}/{len(self.image_files)}\n{name}"
        self.lbl_info.config(text=info)

    def _on_window_resize(self, event):
        if event.widget != self: return
        if self._resize_timer: self.after_cancel(self._resize_timer)
        if hasattr(self, 'canvas'): self._resize_timer = self.after(200, lambda: self.canvas.update_display())

    def _on_mousewheel(self, event):
        if not hasattr(self, 'canvas') or not self.canvas.pil_image: return
        delta = 0; sys = self.tk.call('tk', 'windowingsystem')
        if sys == 'aqua': delta = event.delta * 0.01 # macOS
        elif sys == 'x11': delta = 0.1 if event.num == 4 else (-0.1 if event.num == 5 else 0) # Linux
        else: delta = (event.delta / 120) * 0.1 if hasattr(event, 'delta') else 0 # Windows
        if delta != 0: self.canvas.set_zoom(delta, event)

    def status_update(self, message, timeout=3000, error=False):
        if not hasattr(self, 'lbl_status'): return
        fg = "red" if error else self.style.lookup("TLabel", "foreground"); prefix = "⚠️ " if error else ""
        self.lbl_status.config(text=f" {prefix}{message}", foreground=fg)
        if self._status_clear_timer: self.after_cancel(self._status_clear_timer); self._status_clear_timer = None
        if timeout > 0: self._status_clear_timer = self.after(timeout, lambda: self.lbl_status.config(text="", foreground=self.style.lookup("TLabel", "foreground")))

# --- Main Execution ---
if __name__ == "__main__":
    app = MainApplication()
    app.update_idletasks()
    sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
    ww, wh = app.winfo_width(), app.winfo_height()
    px, py = max(0, (sw//2) - (ww//2)), max(0, (sh//2) - (wh//2))
    app.geometry(f'{ww}x{wh}+{px}+{py}')
    app.mainloop()
