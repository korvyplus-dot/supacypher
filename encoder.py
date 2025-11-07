import tkinter as tk
from tkinter import messagebox
import lzma
import numpy as np

# Define the 64-color palette
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

# Constants for image dimensions and encoding
PHYSICAL_WIDTH = 1200
PHYSICAL_HEIGHT = 600
BLOCK_SIZE = 4 # 4x4 physical pixel block represents one logical pixel
BORDER_SIZE = 20 # Size of the red border
BORDER_COLOR_6BIT_VALUE = 48 # Index for (255, 0, 0) in COLOR_PALETTE
SEPARATOR_VALUE = 63 # Value used for padding and as a separator

# Technical Information Constants
ENCODER_VERSION = 1
ENCODING_CP1251_ID = 0
ENCODING_UTF8_ID = 1 # New ID for UTF-8
COMPRESSOR_LZMA_ID = 1
COMPRESSOR_NONE_ID = 0 # Not used in current implementation, but good to define

# Helper function to convert a byte array to a list of 6-bit integer chunks
def _bytes_to_6bit_chunks(byte_array):
    binary_string = ''.join(bin(byte_val)[2:].zfill(8) for byte_val in byte_array)
    six_bit_chunks = []
    for i in range(0, len(binary_string), 6):
        chunk = binary_string[i:i+6]
        six_bit_chunks.append(int(chunk.ljust(6, '0'), 2))
    return six_bit_chunks

# Generate the 10-byte technical data block
def _generate_technical_data_bytes(encoding_id, compressor_id):
    tech_data = bytearray(10)
    tech_data[0] = ENCODER_VERSION
    tech_data[1] = encoding_id
    tech_data[2] = compressor_id
    # Bytes 3-9 are reserved (default to 0)
    return tech_data

# Generate the full separator as a list of 6-bit chunks
def _generate_full_separator_chunks(encoding_id, compressor_id):
    separator_start_bytes = bytearray([SEPARATOR_VALUE, SEPARATOR_VALUE])
    separator_end_bytes = bytearray([SEPARATOR_VALUE, SEPARATOR_VALUE])
    
    tech_data_bytes = _generate_technical_data_bytes(encoding_id, compressor_id)
    
    full_separator_bytes = separator_start_bytes + tech_data_bytes + separator_end_bytes
    return _bytes_to_6bit_chunks(full_separator_bytes)

# FULL_SEPARATOR_CHUNKS and SEPARATOR_LENGTH will now be generated dynamically in encode_text

class EncoderApp:
    def __init__(self, master):
        self.master = master
        master.title("Supacypher Encoder")

        self.text_input = tk.Text(master, wrap=tk.WORD, width=80, height=10)
        self.text_input.pack(pady=10)

        # Encoding selection
        self.encoding_options = {"CP-1251": ENCODING_CP1251_ID, "UTF-8": ENCODING_UTF8_ID}
        self.selected_encoding_name = tk.StringVar(master)
        self.selected_encoding_name.set("CP-1251") # default value

        encoding_menu = tk.OptionMenu(master, self.selected_encoding_name, *self.encoding_options.keys())
        encoding_menu.pack(pady=5)

        # Compression selection
        self.compressor_options = {"None": COMPRESSOR_NONE_ID, "LZMA": COMPRESSOR_LZMA_ID}
        self.selected_compressor_name = tk.StringVar(master)
        self.selected_compressor_name.set("LZMA") # default value

        compressor_menu = tk.OptionMenu(master, self.selected_compressor_name, *self.compressor_options.keys())
        compressor_menu.pack(pady=5)

        self.encode_button = tk.Button(master, text="Encode Text", command=self.encode_text)
        self.encode_button.pack(pady=5)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        self.tk_image = None # To hold the PhotoImage object

    def encode_text(self):
        input_text = self.text_input.get(1.0, tk.END).strip()

        if not input_text:
            messagebox.showwarning("Warning", "Please enter some text to encode.")
            return

        try:
            # Get selected encoding ID and name
            current_encoding_name = self.selected_encoding_name.get()
            current_encoding_id = self.encoding_options[current_encoding_name]
            
            current_compressor_name = self.selected_compressor_name.get()
            current_compressor_id = self.compressor_options[current_compressor_name]

            # Generate dynamic separator based on selected encoding and compressor
            FULL_SEPARATOR_CHUNKS = _generate_full_separator_chunks(current_encoding_id, current_compressor_id)
            SEPARATOR_LENGTH = len(FULL_SEPARATOR_CHUNKS)

            # 1. Encode text to selected encoding
            encoded_bytes = input_text.encode(current_encoding_name.lower().replace('-', ''))

            # 2. Compress based on selection
            if current_compressor_id == COMPRESSOR_LZMA_ID:
                compressed_bytes = lzma.compress(encoded_bytes)
            elif current_compressor_id == COMPRESSOR_NONE_ID:
                compressed_bytes = encoded_bytes
            else:
                raise ValueError(f"Unknown compressor ID: {current_compressor_id}")

            # 3. Convert compressed bytes to a binary string
            binary_string = ''.join(bin(byte_val)[2:].zfill(8) for byte_val in compressed_bytes)

            # 4. Split binary string into 6-bit chunks and convert to integers
            six_bit_chunks = []
            for i in range(0, len(binary_string), 6):
                chunk = binary_string[i:i+6]
                # Pad with '0' if the last chunk is less than 6 bits
                six_bit_chunks.append(int(chunk.ljust(6, '0'), 2))

            # Calculate logical dimensions
            logical_width = PHYSICAL_WIDTH // BLOCK_SIZE
            logical_height = PHYSICAL_HEIGHT // BLOCK_SIZE
            total_logical_pixels = logical_width * logical_height

            # 5. Prepend technical data and then add padding/redundancy
            # The very first block of data should be the technical information (which is part of the full separator)
            # This ensures the technical data is always at the beginning of the encoded stream.

            technical_data_prefix = _bytes_to_6bit_chunks(_generate_technical_data_bytes(current_encoding_id, current_compressor_id))
            data_with_tech_prefix = technical_data_prefix + six_bit_chunks

            # Calculate logical dimensions
            logical_width = PHYSICAL_WIDTH // BLOCK_SIZE
            logical_height = PHYSICAL_HEIGHT // BLOCK_SIZE
            total_logical_pixels = logical_width * logical_height

            final_six_bit_data = []
            current_data_length = len(data_with_tech_prefix)
            
            # If the combined data (tech prefix + content) is less than total_logical_pixels,
            # we need to add separators and repeat data.
            if current_data_length < total_logical_pixels:
                final_six_bit_data.extend(data_with_tech_prefix)
                
                # Keep adding the full separator and then the original data (with tech prefix)
                # until total_logical_pixels is reached.
                while len(final_six_bit_data) < total_logical_pixels:
                    final_six_bit_data.extend(FULL_SEPARATOR_CHUNKS)
                    if len(final_six_bit_data) < total_logical_pixels: # Only add data if space remains
                        final_six_bit_data.extend(data_with_tech_prefix)
                        
                # Truncate if we've exceeded the total_logical_pixels
                final_six_bit_data = final_six_bit_data[:total_logical_pixels]
            else:
                # If data is too long, truncate it
                final_six_bit_data = data_with_tech_prefix[:total_logical_pixels]
            
            # Ensure the final data length matches total_logical_pixels
            if len(final_six_bit_data) < total_logical_pixels:
                # This case should ideally not happen with the above logic, but as a safeguard
                final_six_bit_data.extend([SEPARATOR_VALUE] * (total_logical_pixels - len(final_six_bit_data)))
            elif len(final_six_bit_data) > total_logical_pixels:
                final_six_bit_data = final_six_bit_data[:total_logical_pixels]

            # 6. Create image array (content only, without border initially)
            image_array = np.zeros((PHYSICAL_HEIGHT, PHYSICAL_WIDTH, 3), dtype=np.uint8)
            
            for i in range(logical_height):
                for j in range(logical_width):
                    data_index = i * logical_width + j
                    if data_index < len(final_six_bit_data):
                        color_index = final_six_bit_data[data_index]
                        color = COLOR_PALETTE[color_index]
                        
                        # Fill the 4x4 block with the color
                        image_array[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = color
                    else:
                        # This part should ideally not be reached if padding is correct
                        image_array[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = COLOR_PALETTE[SEPARATOR_VALUE]


            # 7. Add Red Border
            bordered_width = PHYSICAL_WIDTH + 2 * BORDER_SIZE
            bordered_height = PHYSICAL_HEIGHT + 2 * BORDER_SIZE
            border_color = COLOR_PALETTE[BORDER_COLOR_6BIT_VALUE]

            bordered_image_array = np.full((bordered_height, bordered_width, 3), border_color, dtype=np.uint8)
            
            # Place the content image in the center
            bordered_image_array[BORDER_SIZE:BORDER_SIZE+PHYSICAL_HEIGHT, BORDER_SIZE:BORDER_SIZE+PHYSICAL_WIDTH] = image_array

            # 8. Convert to PPM format and display
            height, width, _ = bordered_image_array.shape
            ppm_header = f'P6\n{width} {height}\n255\n'.encode('ascii')
            ppm_data = bordered_image_array.tobytes()
            
            # Create a PhotoImage from the PPM data
            self.tk_image = tk.PhotoImage(width=width, height=height, data=ppm_header + ppm_data, format='PPM')
            self.image_label.config(image=self.tk_image)
            messagebox.showinfo("Success", "Text encoded into image!")

        except Exception as e:
            messagebox.showerror("Encoding Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EncoderApp(root)
    root.mainloop()