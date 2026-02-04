import lzma
import base64
import sys
import os


def decompress_file(filename):
    """
    Decompress a file that was compressed with lzma after base64 encoding.
    
    Args:
        filename (str): Path to the compressed file
    
    Returns:
        str: Path to the decompressed output file
    """
    # Read the compressed data from the input file
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Decompress the data: base64 decode then lzma decompress
    original = lzma.decompress(base64.b64decode(data)).decode('utf-8')
    
    # Generate output filename
    base_name = os.path.splitext(filename)[0]  # Remove extension
    output_filename = base_name + '_decompressed.txt'
    
    # Write the decompressed data to the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(original)
    
    print(f"Decompressed file saved as: {output_filename}")
    return output_filename


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python con.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        sys.exit(1)
    
    try:
        output_file = decompress_file(filename)
        print(f"Successfully decompressed '{filename}' to '{output_file}'")
    except Exception as e:
        print(f"Error during decompression: {e}")
        sys.exit(1)