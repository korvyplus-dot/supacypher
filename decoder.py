import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageGrab
import numpy as np
import sys
from collections import Counter
import lzma


# Define the 64-color palette (copied from encoder.py)
def generate_palette():
    palette = []
    # 4 levels for each R, G, B channel to get 4*4*4 = 64 distinct colors
    levels = [0, 85, 170, 255] 
    for r in levels:
        for g in levels:
            for b in levels:
                palette.append((r, g, b))
    return palette

COLOR_PALETTE = generate_palette()
# Use the first color in the palette for padding (which is (0,0,0) - black)
PADDING_6BIT_VALUE = 0 

# Image dimensions and block size (copied from encoder.py)
PHYSICAL_WIDTH = 1200
PHYSICAL_HEIGHT = 600
BLOCK_SIZE = 4 # 4x4 physical pixel block represents one logical pixel
BORDER_COLOR_6BIT_VALUE = 48 # Index for (255, 0, 0) in COLOR_PALETTE
SEPARATOR_VALUE = 63
SEPARATOR_LENGTH = 4

def six_bit_chunks_to_bytes(six_bit_chunks):
    """Converts a sequence of 6-bit chunks back into 8-bit bytes."""
    bit_stream = ""
    for chunk_val in six_bit_chunks:
        bit_stream += bin(chunk_val)[2:].zfill(6)

    # Truncate any partial 6-bit chunk at the end that was used for padding
    # We only care about full 8-bit bytes
    
    byte_stream = []
    for i in range(0, len(bit_stream), 8):
        byte_chunk = bit_stream[i:i+8]
        if len(byte_chunk) == 8: # Only process full bytes
            byte_stream.append(int(byte_chunk, 2))
    return bytes(byte_stream)

def find_nearest_color_index(target_color, palette):
    """Finds the index of the nearest color in the palette to the target_color."""
    target_color_np = np.array(target_color)
    distances = np.sqrt(np.sum((palette - target_color_np)**2, axis=1))
    return np.argmin(distances)

def detect_block_size(img_array, expected_width, expected_height):
    """Infers the block size (width and height) from the image by analyzing color transitions."""
    if expected_height < 2 or expected_width < 2: # Need at least 2x2 to detect transitions
        raise ValueError("Image too small to detect block size.")

    # Detect horizontal block size (block_width)
    scan_row = img_array[expected_height // 2, :, :3]
    change_points_x = [0]
    for i in range(1, expected_width):
        if not np.array_equal(scan_row[i], scan_row[i-1]):
            change_points_x.append(i)
    change_points_x.append(expected_width)

    block_widths = []
    for i in range(1, len(change_points_x)):
        size = change_points_x[i] - change_points_x[i-1]
        if size > 0:
            block_widths.append(size)
    
    if not block_widths:
        raise ValueError("Could not detect any block width from horizontal scan.")
    inferred_block_width = Counter(block_widths).most_common(1)[0][0]

    # Detect vertical block size (block_height)
    scan_col = img_array[:, expected_width // 2, :3]
    change_points_y = [0]
    for i in range(1, expected_height):
        if not np.array_equal(scan_col[i], scan_col[i-1]):
            change_points_y.append(i)
    change_points_y.append(expected_height)

    block_heights = []
    for i in range(1, len(change_points_y)):
        size = change_points_y[i] - change_points_y[i-1]
        if size > 0:
            block_heights.append(size)

    if not block_heights:
        raise ValueError("Could not detect any block height from vertical scan.")
    inferred_block_height = Counter(block_heights).most_common(1)[0][0]

    # Basic validation
    if not isinstance(inferred_block_width, int) or inferred_block_width <= 0:
        raise ValueError(f"Inferred block width is invalid: {inferred_block_width}")
    if not isinstance(inferred_block_height, int) or inferred_block_height <= 0:
        raise ValueError(f"Inferred block height is invalid: {inferred_block_height}")
    
    if expected_width % inferred_block_width != 0 or expected_height % inferred_block_height != 0:
        print(f"Warning: Inferred block size ({inferred_block_width}x{inferred_block_height}) does not evenly divide image dimensions ({expected_width}x{expected_height}). This might lead to decoding errors.", file=sys.stderr)

    return inferred_block_width, inferred_block_height

def decode_image(image):
    """
    Decodes text from an image that was encoded using the encoder.py logic.
    """
    img_array = np.array(image)

    expected_border_color = np.array(COLOR_PALETTE[BORDER_COLOR_6BIT_VALUE])
    
    # Find all pixels that match the expected border color
    # This creates a boolean mask where True indicates a border pixel
    border_pixel_mask = np.all(img_array[:, :, :3] == expected_border_color, axis=2)

    # Get the coordinates of all border pixels
    border_pixel_coords = np.argwhere(border_pixel_mask)

    if border_pixel_coords.size == 0:
        raise ValueError("No border pixels found in the image. Is the image encoded correctly?")

    # Find the bounding box of these border pixels
    min_y, min_x = border_pixel_coords.min(axis=0)[:2] # Top-left corner of the border
    max_y, max_x = border_pixel_coords.max(axis=0)[:2] # Bottom-right corner of the border

    print(f"Debug: min_y={min_y}, min_x={min_x}, max_y={max_y}, max_x={max_x}", file=sys.stderr)

    # The detected bounding box should correspond to the outer edges of the red frame.
    # The actual encoded content starts BORDER_SIZE pixels *inside* this bounding box.
    # Dynamically calculate the scaled border size
    original_border_size = 20 # Original border size from encoder.py
    detected_border_width = max_x - min_x + 1
    detected_border_height = max_y - min_y + 1

    # Calculate scaling factors based on the original bordered image dimensions
    original_bordered_width = PHYSICAL_WIDTH + 2 * original_border_size
    original_bordered_height = PHYSICAL_HEIGHT + 2 * original_border_size

    scale_x = detected_border_width / original_bordered_width
    scale_y = detected_border_height / original_bordered_height

    # Use the average scale or assume uniform scaling
    # For simplicity, let's assume uniform scaling and use scale_x
    scaled_border_size = int(original_border_size * scale_x)

    print(f"Debug: Detected border width: {detected_border_width}, height: {detected_border_height}", file=sys.stderr)
    print(f"Debug: Original bordered width: {original_bordered_width}, height: {original_bordered_height}", file=sys.stderr)
    print(f"Debug: Scale X: {scale_x}, Scale Y: {scale_y}", file=sys.stderr)
    print(f"Debug: Scaled border size: {scaled_border_size}", file=sys.stderr)

    content_start_y = min_y + scaled_border_size
    content_end_y = max_y - scaled_border_size + 1 # +1 because slicing is exclusive
    content_start_x = min_x + scaled_border_size
    content_end_x = max_x - scaled_border_size + 1 # +1 because slicing is exclusive

    # Crop the image to the content area (excluding the border)
    cropped_img_array_raw = img_array[content_start_y:content_end_y, content_start_x:content_end_x, :3]
    print(f"Debug: cropped_img_array_raw shape: {cropped_img_array_raw.shape}", file=sys.stderr)

    # Convert the numpy array back to a PIL Image for resizing
    cropped_pil_image = Image.fromarray(cropped_img_array_raw)

    # Resize the image back to the expected physical dimensions
    # This handles cases where the image might have been scaled after encoding
    resized_pil_image = cropped_pil_image.resize((PHYSICAL_WIDTH, PHYSICAL_HEIGHT), Image.Resampling.NEAREST)

    # Convert back to numpy array for further processing
    cropped_img_array = np.array(resized_pil_image)

    # Debug: Print unique colors in the cropped_img_array
    unique_colors = np.unique(cropped_img_array.reshape(-1, cropped_img_array.shape[2]), axis=0)
    print(f"Debug: Unique colors in cropped_img_array: {unique_colors}", file=sys.stderr)

    # Verify cropped dimensions after resizing
    if cropped_img_array.shape[0] != PHYSICAL_HEIGHT or cropped_img_array.shape[1] != PHYSICAL_WIDTH:
        # This check should ideally pass after resizing, but good for sanity
        raise ValueError(f"Resized image dimensions ({cropped_img_array.shape[1]}x{cropped_img_array.shape[0]}) do not match expected physical dimensions ({PHYSICAL_WIDTH}x{PHYSICAL_HEIGHT}). This indicates an issue with resizing.")

    # Dynamically detect BLOCK_SIZE
    inferred_block_width, inferred_block_height = detect_block_size(cropped_img_array, PHYSICAL_WIDTH, PHYSICAL_HEIGHT)
    print(f"Debug: Inferred block size: {inferred_block_width}x{inferred_block_height}", file=sys.stderr)
    
    logical_width = PHYSICAL_WIDTH // inferred_block_width
    logical_height = PHYSICAL_HEIGHT // inferred_block_height

    palette_np = np.array(COLOR_PALETTE, dtype=np.uint8)

    six_bit_chunks = []
    for i in range(logical_height):
        for j in range(logical_width):
            # Extract the block using the inferred block size
            block = cropped_img_array[i*inferred_block_height:(i+1)*inferred_block_height, j*inferred_block_width:(j+1)*inferred_block_width]
            
            # Find the most frequent color in the block
            colors_in_block = block.reshape(-1, 3)
            color_counts = Counter(map(tuple, colors_in_block))
            pixel_color_tuple = color_counts.most_common(1)[0][0]
            pixel_color_np = np.array(pixel_color_tuple, dtype=np.uint8)

            # Instead of finding the nearest color, we will look for an exact match.
            # If no exact match is found, it indicates an issue with the image or encoding.
            
            # Convert the pixel_color_np to a tuple to compare with COLOR_PALETTE (which is a list of tuples)
            
            # Find the nearest color in the palette
            six_bit_value = find_nearest_color_index(pixel_color_np, palette_np)
            six_bit_chunks.append(six_bit_value)

    print(f"First 20 six_bit_chunks: {six_bit_chunks[:20]}", file=sys.stderr)
    print(f"Last 20 six_bit_chunks: {six_bit_chunks[-20:]}", file=sys.stderr)
    print(f"Length of six_bit_chunks: {len(six_bit_chunks)}", file=sys.stderr)

    # Find the separator sequence to determine the actual end of data
    try:
        # The separator is a sequence of SEPARATOR_LENGTH (4) instances of SEPARATOR_VALUE (63)
        separator_index = -1
        for i in range(len(six_bit_chunks) - SEPARATOR_LENGTH + 1):
            if all(six_bit_chunks[i + k] == SEPARATOR_VALUE for k in range(SEPARATOR_LENGTH)):
                separator_index = i
                break
        
        if separator_index != -1:
            # Data is everything before the separator
            data_six_bit_chunks = six_bit_chunks[:separator_index]
        else:
            # If no separator is found, it means the data filled the entire image
            # or the separator was truncated. In this case, we assume all chunks are data.
            data_six_bit_chunks = six_bit_chunks
            print("Warning: Separator not found. Assuming all data is original content.", file=sys.stderr)

    except Exception as e:
        raise ValueError(f"Error finding separator: {e}")

    # Convert 6-bit chunks back to bytes
    decoded_bytes = six_bit_chunks_to_bytes(data_six_bit_chunks)

    # Decompress bytes using lzma
    try:
        decompressed_bytes = lzma.decompress(decoded_bytes)
    except lzma.LZMAError as e:
        raise ValueError(f"LZMA decompression error: {e}. Data might be corrupted or not LZMA compressed.")

    # Decode bytes to text using cp1251
    try:
        decoded_text = decompressed_bytes.decode('cp1251')
    except UnicodeDecodeError:
        raise ValueError("Could not decode bytes to text using CP-1251. Data might be corrupted or not CP-1251 encoded.")

    print(f"Value of decoded_text before return: {repr(decoded_text)}") # New debug line
    return decoded_text


class DecoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Decoder")
        self.root.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        decode_button = tk.Button(button_frame, text="Decode from Clipboard", command=self.decode_from_clipboard)
        decode_button.pack(side=tk.LEFT, padx=5)



        # Text area for decoded output
        self.decoded_text_widget = tk.Text(self.root, wrap=tk.WORD, width=90, height=30)
        self.decoded_text_widget.pack(pady=10)

    def decode_from_clipboard(self):
        self.decoded_text_widget.delete(1.0, tk.END) # Clear previous text
        try:
            image = ImageGrab.grabclipboard()
            if image is None:
                messagebox.showerror("Error", "No image found in clipboard.")
                return
            
            # Ensure the image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')

            decoded_text = decode_image(image)
            print(f"Decoded text (repr): {repr(decoded_text)}") # Debugging line
            self.decoded_text_widget.insert(tk.END, decoded_text)
            messagebox.showinfo("Success", "Image decoded successfully!")

        except ValueError as ve:
            messagebox.showerror("Decoding Error", str(ve))
        except Exception as e:
            messagebox.showerror("An unexpected error occurred", str(e))



if __name__ == "__main__":
    root = tk.Tk()
    app = DecoderApp(root)
    root.mainloop()
