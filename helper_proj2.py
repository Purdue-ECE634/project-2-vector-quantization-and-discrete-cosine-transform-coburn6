import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os
from scipy.fftpack import dct, idct


class VectorQuantization:
    def __init__(self, block_size=4):
        """
        Constructor for the Vector Quantization compressor.

        Inputs:
        -----------
        block_size : Size of the square blocks to use for vector quantization.
        """
        self.block_size = block_size
        self.codebook = None
        self.kmeans = None

    def extract_blocks(self, image):
        """
        Extract blocks of size block_size x block_size from the image.

        Inputs:
        -----------
        image: Input grayscale image.

        Returns:
        --------
        Array of blocks flattened to vectors.
        """
        height, width = image.shape

        # Calculate the number of blocks in each dimension
        n_blocks_h = height // self.block_size
        n_blocks_w = width // self.block_size

        # Truncate the image to fit complete blocks (needed for edge cases)
        truncated_h = n_blocks_h * self.block_size
        truncated_w = n_blocks_w * self.block_size
        truncated_image = image[:truncated_h, :truncated_w]

        # Extract blocks and reshape them to vectors
        blocks = np.zeros(
            (n_blocks_h * n_blocks_w, self.block_size**2), dtype=np.float32
        )

        # Extract each block and place in the corresponding row of the 'blocks' array
        block_idx = 0
        for i in range(0, truncated_h, self.block_size):
            for j in range(0, truncated_w, self.block_size):
                block = truncated_image[
                    i : i + self.block_size, j : j + self.block_size
                ]
                blocks[block_idx, :] = block.flatten()
                block_idx += 1

        return blocks

    def initialize_codebook(self, training_vectors, codebook_size):
        """
        Initialize the codebook for GLA.

        Inputs:
        -----------
        training_vectors : Array of training vectors.
        codebook_size : Size of the codebook (number of codewords).

        Returns:
        --------
        numpy.ndarray
            Initial codebook.
        """
        # Random selection from training vectors
        indices = np.random.choice(
            len(training_vectors), size=codebook_size, replace=False
        )
        return training_vectors[indices].copy()

    def find_nearest_codeword(self, vector, codebook):
        """
        Find the index of the nearest codeword in the codebook for a given vector.

        Inputs:
        -----------
        vector : Input vector.
        codebook : Codebook of codewords (in array format).

        Returns:
        --------
        int
            Index of the nearest codeword.
        """
        distances = np.sum((codebook - vector) ** 2, axis=1)
        return np.argmin(distances)

    def update_codebook(self, training_vectors, codeword_assignments, codebook_size):
        """
        Update the codebook based on the current assignments.

        Inputs:
        -----------
        training_vectors : Array of training vectors.
        codeword_assignments : Array of codeword assignments for each training vector.
        codebook_size : Size of the codebook.

        Returns:
        --------
        numpy.ndarray
            Updated codebook.
        bool
            Whether any codeword was updated or not.
        """
        # Initialize a new codebook
        new_codebook = np.zeros(
            (codebook_size, training_vectors.shape[1]), dtype=np.float32
        )
        updated = False

        # Iterate through our codewords
        for i in range(codebook_size):
            # Find all vectors assigned to this codeword
            indices = np.where(codeword_assignments == i)[0]

            if len(indices) > 0:
                # Update codeword to be the centroid of its assigned vectors
                new_codeword = np.mean(training_vectors[indices], axis=0)

                # Check if the codeword has changed
                if not np.array_equal(new_codeword, self.codebook[i]):
                    updated = True

                new_codebook[i] = new_codeword
            else:
                # If no vectors assigned, keep the old codeword
                new_codebook[i] = self.codebook[i]

        return new_codebook, updated

    def calculate_distortion(self, training_vectors, assignments):
        """
        Calculate the average distortion for the current codebook.

        Inputs:
        -----------
        training_vectors : Array of training vectors.
        assignments : Array of codeword assignments for each training vector.

        Returns:
        --------
        float
            Average distortion (MSE).
        """
        distortion = 0.0

        # Calculate the distortion for each vector
        for i, vector in enumerate(training_vectors):
            codeword_idx = assignments[i]
            codeword = self.codebook[codeword_idx]
            distortion += np.sum((vector - codeword) ** 2)

        return distortion / len(training_vectors)

    def generalized_lloyd_algorithm(
        self, training_vectors, codebook_size, max_iterations=100, tolerance=1e-6
    ):
        """
        Implement the Generalized Lloyd Algorithm (GLA) for vector quantization.

        Inputs:
        -----------
        training_vectors : Array of training vectors.
        codebook_size : Size of the codebook (number of codewords).
        max_iterations : Maximum number of iterations.
        tolerance : Convergence tolerance for distortion.

        Returns:
        --------
        numpy.ndarray
            The trained codebook.
        list
            List of distortion values per iteration.
        """
        # Initialize codebook
        self.codebook = self.initialize_codebook(training_vectors, codebook_size)

        # Track distortion history
        distortion_history = []
        prev_distortion = float("inf")

        # Iterate until convergence (stops after max_iterations if not converged)
        for iteration in range(max_iterations):
            # Step 1: Find nearest codeword for each training vector
            assignments = np.zeros(len(training_vectors), dtype=int)
            for i, vector in enumerate(training_vectors):
                assignments[i] = self.find_nearest_codeword(vector, self.codebook)

            # Step 2: Update codebook
            self.codebook, updated = self.update_codebook(
                training_vectors, assignments, codebook_size
            )

            # Calculate distortion
            distortion = self.calculate_distortion(training_vectors, assignments)
            distortion_history.append(distortion)

            # Check convergence
            if not updated or abs(prev_distortion - distortion) < tolerance:
                print(f"GLA converged after {iteration+1} iterations.")
                break

            prev_distortion = distortion

            # Handy printout every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"GLA Iteration {iteration+1}: Distortion = {distortion:.4f}")

        # Print a message if we reach the maximum iterations
        if iteration == max_iterations - 1:
            print(f"GLA reached maximum iterations ({max_iterations}).")

        return self.codebook, distortion_history

    def train_codebook(self, images, codebook_size):
        """
        Train the codebook using either GLA or KMeans.

        Inputs:
        -----------
        images : List of images or a single image to train on.
        codebook_size : Size of the codebook (number of codewords).

        Returns:
        --------
        numpy.ndarray
            The trained codebook.
        """
        if not isinstance(images, list):
            images = [images]

        # Extract blocks from all training images
        all_blocks = []
        for img in images:
            blocks = self.extract_blocks(img)
            all_blocks.append(blocks)

        # Stack all blocks into a single array
        training_vectors = np.vstack(all_blocks)

        # Train the codebook 
        print(
            f"Using Generalized Lloyd Algorithm for codebook size {codebook_size}"
        )
        self.codebook, distortion_history = self.generalized_lloyd_algorithm(
            training_vectors, codebook_size
        )

        # Create a basic kmeans object to use for compress() if needed
        from sklearn.cluster import KMeans

        self.kmeans = KMeans(n_clusters=codebook_size)
        self.kmeans.cluster_centers_ = self.codebook
        

        return self.codebook

    def compress(self, image):
        """
        Compress an image using the trained codebook.

        Inputs:
        -----------
        image : Input grayscale image.

        Returns:
        --------
        numpy.ndarray
            Array of indices representing the quantized image.
        compressed_shape : tuple
            Original shape information needed for reconstruction.
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")

        height, width = image.shape

        # Calculate the number of blocks in each dimension
        n_blocks_h = height // self.block_size
        n_blocks_w = width // self.block_size

        # Truncate the image to fit complete blocks
        truncated_h = n_blocks_h * self.block_size
        truncated_w = n_blocks_w * self.block_size

        # Extract blocks
        extracted_blocks = self.extract_blocks(image)

        # Quantize each block to the nearest codeword
        quantized_indices = np.zeros(len(extracted_blocks), dtype=int)
        for i, block in enumerate(extracted_blocks):
            quantized_indices[i] = self.find_nearest_codeword(block, self.codebook)

        # Return the quantized indices and the shape information
        compressed_shape = (n_blocks_h, n_blocks_w)
        return quantized_indices, compressed_shape

    def decompress(self, quantized_indices, compressed_shape):
        """
        Decompress the quantized indices back to an image.

        Inputs:
        -----------
        quantized_indices : Array of indices representing the quantized image.
        compressed_shape : Original shape information from compression.

        Returns:
        --------
        numpy.ndarray
            Reconstructed image.
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")

        n_blocks_h, n_blocks_w = compressed_shape

        # Reconstruct the image
        reconstructed_img = np.zeros(
            (n_blocks_h * self.block_size, n_blocks_w * self.block_size),
            dtype=np.float32,
        )

        # Reconstruct each block from the codeword
        block_idx = 0
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # Get the codeword for this block
                codeword = self.codebook[quantized_indices[block_idx]]

                # Reshape the codeword to a block
                block = codeword.reshape(self.block_size, self.block_size)

                # Place the block in the reconstructed image
                reconstructed_img[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ] = block

                block_idx += 1

        return reconstructed_img

    def save_codebook(self, filepath):
        """
        Save the codebook to a file. Primarily used for debug.

        Inputs:
        -----------
        filepath : Path to save the codebook.
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")

        # Save the codebook to a numpy file
        np.save(filepath, self.codebook)

    def load_codebook(self, filepath):
        """
        Load a codebook from a file. Primarily used for debug.

        Inputs:
        -----------
        filepath : Path to the saved codebook.
        """
        self.codebook = np.load(filepath)

    def calculate_mse(self, original_img, reconstructed_img):
        """
        Calculate the Mean Squared Error between original and reconstructed images.

        Inputs:
        -----------
        original_img : Original image.
        reconstructed_img : Reconstructed image.

        Returns:
        --------
        float
            Mean Squared Error.
        """
        # Ensure both images have the same shape
        h, w = reconstructed_img.shape
        return np.mean((original_img[:h, :w] - reconstructed_img) ** 2)

    def calculate_psnr(self, original_img, reconstructed_img):
        """
        Calculate the Peak Signal-to-Noise Ratio between original and reconstructed images.

        Inputs:
        -----------
        original : Original image.
        reconstructed : Reconstructed image.

        Returns:
        --------
        float
            PSNR in decibels.
        """
        mse = self.calculate_mse(original_img, reconstructed_img)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0  # For 8-bit images
        psnr = 10 * np.log10((max_pixel**2) / mse)
        return psnr

    def compression_ratio(self, original_img, codebook_size):
        """
        Calculate the compression ratio.

        Inputs:
        -----------
        original_img : Original image.
        codebook_size : Size of the codebook.

        Returns:
        --------
        float
            Compression ratio.
        """
        # Original size: 8 bits per pixel
        original_bits = original_img.size * 8

        # Compressed size: log2(codebook_size) bits per block + codebook storage
        bits_per_index = np.ceil(np.log2(codebook_size))
        num_blocks = (original_img.shape[0] // self.block_size) * (
            original_img.shape[1] // self.block_size
        )
        indices_bits = (
            num_blocks * bits_per_index
        )  # Indices: log2(codebook_size) bits per block

        # Codebook storage: codebook_size entries, each entry is block_size^2 pixels, 8 bits per pixel
        codebook_bits = codebook_size * (self.block_size**2) * 8

        # Total compressed bits
        compressed_bits = indices_bits + codebook_bits

        # Compression ratio
        return original_bits / compressed_bits


class DCTCompression:
    def __init__(self, block_size=8):
        """
        Initialize the DCT compressor.

        Inputs:
        -----------
        block_size : Size of the square blocks to use for DCT (defaults to 8 per project instructions).
        """
        self.block_size = block_size

    def dct2(self, block):
        """
        Apply 2D DCT to a block.

        Inputs:
        -----------
        block : Input block of size block_size x block_size.

        Returns:
        --------
        numpy.ndarray
            DCT coefficients.
        """
        return dct(
            dct(block.T, norm="ortho").T, norm="ortho"
        )  # Apply DCT twice, first to columns, then to rows

    def idct2(self, block):
        """
        Apply 2D inverse DCT to a block.

        Inputs:
        -----------
        block : numpy.ndarray
            Input DCT coefficients.

        Returns:
        --------
        numpy.ndarray
            Reconstructed block.
        """
        return idct(
            idct(block.T, norm="ortho").T, norm="ortho"
        )  # Apply inverse DCT twice, first to columns, then to rows

    def extract_blocks(self, image):
        """
        Extract blocks of size block_size x block_size from the image.

        Inputs:
        -----------
        image : numpy.ndarray
            Input grayscale image.

        Returns:
        --------
        numpy.ndarray
            Array of blocks.
        tuple
            Shape information for reconstruction.
        """
        height, width = image.shape

        # Calculate the number of blocks in each dimension
        n_blocks_h = height // self.block_size
        n_blocks_w = width // self.block_size

        # Truncate the image to fit complete blocks
        truncated_h = n_blocks_h * self.block_size
        truncated_w = n_blocks_w * self.block_size
        truncated_image = image[:truncated_h, :truncated_w]

        # Extract blocks
        blocks = np.zeros(
            (n_blocks_h, n_blocks_w, self.block_size, self.block_size), dtype=np.float32
        )

        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                blocks[i, j] = truncated_image[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ]

        return blocks, (n_blocks_h, n_blocks_w)

    def zigzag_indices(self, size):
        """
        Generate indices for zigzag traversal of a square matrix.

        Inputs:
        -----------
        size : Size of the square matrix.

        Returns:
        --------
        list
            List of indices (row, col) in zigzag order.
        """
        indices = []
        # Traverse the first row
        for sum_idx in range(2 * size - 1):
            # Even sum indices go up and right
            if sum_idx % 2 == 0:
                start_i = min(sum_idx, size - 1)
                end_i = max(0, sum_idx - size + 1)
                # Traverse the diagonal
                for i in range(start_i, end_i - 1, -1):
                    j = sum_idx - i
                    indices.append((i, j))
            else:  # Odd - go down and left
                start_j = min(sum_idx, size - 1)
                end_j = max(0, sum_idx - size + 1)
                # Traverse the diagonal
                for j in range(start_j, end_j - 1, -1):
                    i = sum_idx - j
                    indices.append((i, j))
        return indices

    def compress(self, image, k):
        """
        Compress an image using DCT and keep only the first k coefficients in zigzag order.

        Inputs:
        -----------
        image : Input grayscale image.
        k : Number of DCT coefficients to keep per block.

        Returns:
        --------
        numpy.ndarray
            Compressed DCT coefficients.
        tuple
            Shape information for reconstruction.
        """
        # Extract blocks
        blocks, shape_info = self.extract_blocks(image)
        n_blocks_h, n_blocks_w = shape_info

        # Transform each block using DCT
        dct_blocks = np.zeros_like(blocks)
        # Apply DCT to each block
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                dct_blocks[i, j] = self.dct2(blocks[i, j])

        # Generate zigzag traversal indices
        zigzag_idx = self.zigzag_indices(self.block_size)

        # Threshold DCT coefficients to keep only the first k coefficients in zigzag order
        compressed_blocks = np.zeros_like(dct_blocks)
        # Apply thresholding to each block
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # Get the DCT coefficients
                dct_block = dct_blocks[i, j]

                # Create a mask using the first k indices in zigzag order
                mask = np.zeros((self.block_size, self.block_size), dtype=bool)
                # Set the mask to True for the first k indices
                for idx in range(min(k, len(zigzag_idx))):
                    row, col = zigzag_idx[idx]
                    mask[row, col] = True

                # Apply the mask
                compressed_blocks[i, j] = dct_block * mask

        return compressed_blocks, shape_info

    def decompress(self, compressed_blocks, shape_info):
        """
        Decompress the DCT coefficients back to an image.

        Inputs:
        -----------
        compressed_blocks : Compressed DCT coefficients.
        shape_info : Shape information from compression.

        Returns:
        --------
        numpy.ndarray
            Reconstructed image.
        """
        n_blocks_h, n_blocks_w = shape_info

        # Reconstruct the image
        reconstructed_img = np.zeros(
            (n_blocks_h * self.block_size, n_blocks_w * self.block_size),
            dtype=np.float32,
        )

        # Apply inverse DCT to each block
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # Get the compressed DCT coefficients
                dct_block = compressed_blocks[i, j]

                # Apply inverse DCT
                block = self.idct2(dct_block)

                # Place the block in the reconstructed image
                reconstructed_img[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ] = block

        return reconstructed_img

    def calculate_mse(self, original_img, reconstructed_img):
        """
        Calculate the Mean Squared Error between original and reconstructed images.

        Inputs:
        -----------
        original_img: Original image.
        reconstructed_img : Reconstructed image.

        Returns:
        --------
        float
            Mean Squared Error.
        """
        # Ensure both images have the same shape
        h, w = reconstructed_img.shape
        return np.mean((original_img[:h, :w] - reconstructed_img) ** 2)

    def calculate_psnr(self, original_img, reconstructed_img):
        """
        Calculate the Peak Signal-to-Noise Ratio between original and reconstructed images.

        Inputs:
        -----------
        original_img : Original image.
        reconstructed_img : Reconstructed image.

        Returns:
        --------
        float
            PSNR in decibels.
        """
        mse = self.calculate_mse(original_img, reconstructed_img)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0  # Assuming 8-bit images
        psnr = 10 * np.log10((max_pixel**2) / mse)
        return psnr

    def compression_ratio(self, original_img, K):
        """
        Calculate the compression ratio.

        Inputs:
        -----------
        original : Original image.
        K : Number of DCT coefficients kept per block.

        Returns:
        --------
        float
            Compression ratio.
        """
        # Original: 8 bits per pixel
        original_bits = original_img.size * 8

        # Compressed: k coefficients per block, each coefficient is stored with value and position
        num_blocks = (original_img.shape[0] // self.block_size) * (
            original_img.shape[1] // self.block_size
        )
        compressed_bits = num_blocks * K * (8 + np.ceil(np.log2(self.block_size**2)))

        return original_bits / compressed_bits


def load_images(image_paths):
    """
    Load grayscale images from paths.

    Inputs:
    -----------
    image_paths : List of paths to images.

    Returns:
    --------
    list
        List of grayscale images as numpy arrays.
    """
    images = []
    # Load each image and automatically convert to grayscale
    for path in image_paths:
        img = Image.open(path).convert("L")
        images.append(np.array(img))
    return images


def run_vector_quantization_experiment(
    test_images, training_images=None, codebook_sizes=[128, 256], block_size=4
):
    """
    Run vector quantization experiments with different codebook sizes.

    Inputs:
    -----------
    test_images : List of test images as numpy arrays.
    training_images : List of training images as numpy arrays. If None, use test_images for training.
    codebook_sizes : List of codebook sizes to test.
    block_size : Size of the blocks to use.

    Returns:
    --------
    dict
        Dictionary containing experiment results.
    """
    if training_images is None:
        training_images = test_images

    results = {}

    # Train and test for each codebook size
    for codebook_size in codebook_sizes:
        vq = VectorQuantization(block_size=block_size)
        vq.train_codebook(training_images, codebook_size)

        test_results = []
        # Test on each image
        for i, test_img in enumerate(test_images):
            # Compress and decompress
            quantized_indices, compressed_shape = vq.compress(test_img)
            reconstructed = vq.decompress(quantized_indices, compressed_shape)

            # Calculate metrics
            mse = vq.calculate_mse(test_img, reconstructed)
            psnr = vq.calculate_psnr(test_img, reconstructed)
            cr = vq.compression_ratio(test_img, codebook_size)

            test_results.append(
                {
                    "image_index": i,
                    "mse": mse,
                    "psnr": psnr,
                    "compression_ratio": cr,
                    "reconstructed": reconstructed,
                }
            )

        results[codebook_size] = test_results

    return results


def display_results(test_images, results, title_prefix=""):
    """
    Display original and reconstructed images along with metrics.

    Inputs:
    -----------
    test_images : List of original test images.
    results : Results dictionary from run_vector_quantization_experiment.
    title_prefix : Prefix for plot titles.
    """

    # Display original and reconstructed images
    for codebook_size, test_results in results.items():
        for result in test_results:
            img_idx = result["image_index"]
            original_img = test_images[img_idx]
            reconstructed_img = result["reconstructed"]

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Display original
            axes[0].imshow(original_img, cmap="gray")
            axes[0].set_title(f"Original Image {img_idx+1}")
            axes[0].axis("off")

            # Display reconstructed
            axes[1].imshow(reconstructed_img, cmap="gray")
            axes[1].set_title(
                f'{title_prefix}Reconstructed (Codebook Size={codebook_size})\nMSE={result["mse"]:.2f}, PSNR={result["psnr"]:.2f} dB, CR={result["compression_ratio"]:.2f}x'
            )
            axes[1].axis("off")

            plt.tight_layout()
            print(
                f"Saving {title_prefix}reconstructed_{img_idx+1}_codebook_{codebook_size}.png..."
            )
            plt.savefig(
                f"{title_prefix}reconstructed_{img_idx+1}_codebook_{codebook_size}.png"
            )
            plt.show()


def load_image(image_path):
    """
    Load a grayscale image from a path.

    Inputs:
    -----------
    image_path : Path to the image.

    Returns:
    --------
    numpy.ndarray
        Grayscale image as a numpy array.
    """
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)


def run_dct_experiment(image_paths, K_values=[2, 4, 8, 16, 32], block_size=8):
    """
    Run DCT compression experiments with different numbers of coefficients.

    Inputs:
    -----------
    image_paths : List of paths to test images.
    K_values : List of numbers of coefficients to keep.
    block_size : Size of the blocks to use.

    Returns:
    --------
    dict
        Dictionary containing experiment results.
    """
    results = {}

    # Run experiments for each image
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image = load_image(image_path)

        image_results = {}
        dct = DCTCompression(block_size=block_size)  # Initialize DCT compressor

        # Run experiments for each K value
        for k in K_values:
            # Compress and decompress
            compressed, shape_info = dct.compress(image, k)
            reconstructed = dct.decompress(compressed, shape_info)

            # Calculate metrics
            mse = dct.calculate_mse(image, reconstructed)
            psnr = dct.calculate_psnr(image, reconstructed)
            cr = dct.compression_ratio(image, k)

            # Store results
            image_results[k] = {
                "reconstructed": reconstructed,
                "mse": mse,
                "psnr": psnr,
                "compression_ratio": cr,
            }

        results[image_name] = image_results

    return results


def display_dct_results(image_paths, results, output_dir="output_plots"):
    """
    Display original and reconstructed images along with metrics, and save the plots.

    Inputs:
    -----------
    image_paths : List of paths to test images.
    results : Results dictionary from run_dct_experiment.
    output_dir : Directory to save output plots.
    """
    # Create output directory if it doesn't exist
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Store PSNR data for combined plot
    all_psnr_data = {}
    k_values = None

    # Display and save results for each image
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]  # Get name without extension
        image = load_image(image_path)

        image_results = results[image_name]
        if k_values is None:
            k_values = sorted(image_results.keys())

        # Store PSNR values for this image
        all_psnr_data[image_name] = [image_results[k]["psnr"] for k in k_values]

        # Display and save original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        plt.title(f"Original Image: {image_name}")
        plt.axis("off")
        print(f"Saving original image plot to {output_dir}/{base_name}_original.png")
        plt.savefig(f"{output_dir}/{base_name}_original.png", dpi=300)
        plt.close()

        # Display and save the reconstructed images (all K values on one figure)
        cols = min(3, len(k_values))
        rows = (len(k_values) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

        # Plot reconstructed images for each K value
        for i, k in enumerate(k_values):
            result = image_results[k]
            reconstructed = result["reconstructed"]

            axes[i].imshow(reconstructed, cmap="gray")
            axes[i].set_title(
                f'K={k}, PSNR={result["psnr"]:.2f} dB\nMSE={result["mse"]:.2f}, CR={result["compression_ratio"]:.2f}x'
            )
            axes[i].axis("off")

        # Hide empty subplots
        for i in range(len(k_values), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        print(
            f"Saving reconstructed images plot to {output_dir}/{base_name}_reconstructed.png"
        )
        plt.savefig(f"{output_dir}/{base_name}_reconstructed.png", dpi=300)
        plt.close()

        # Save individual PSNR vs K plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, all_psnr_data[image_name], "o-")
        plt.xlabel("Number of DCT Coefficients (K)")
        plt.ylabel("PSNR (dB)")
        plt.title(f"PSNR vs K for {image_name}")
        plt.grid(True)
        print(f"Saving PSNR vs K plot to {output_dir}/{base_name}_psnr_curve.png")
        plt.savefig(f"{output_dir}/{base_name}_psnr_curve.png", dpi=300)
        plt.close()

    # Create and save combined plot with all PSNR curves
    plt.figure(figsize=(12, 8))

    # Define markers and colors for each image
    markers = ["o", "s", "^", "D", "x", "*", "p", "v", "<", ">"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]

    # Plot PSNR curves for all images
    for i, (image_name, psnr_values) in enumerate(all_psnr_data.items()):
        marker_idx = i % len(markers)
        color_idx = i % len(colors)
        plt.plot(
            k_values,
            psnr_values,
            marker=markers[marker_idx],
            color=colors[color_idx],
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=image_name,
        )

    plt.xlabel("Number of DCT Coefficients (K)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("PSNR vs K for All Images", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best")

    # Add horizontal line for PSNR threshold if defined
    if "psnr_threshold" in globals():
        plt.axhline(
            y=psnr_threshold,
            color="r",
            linestyle="--",
            label=f"PSNR Threshold ({psnr_threshold} dB)",
        )
        # Update legend to include the threshold line
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc="best")

    plt.tight_layout()
    print(f"Saving combined PSNR plot to {output_dir}/combined_psnr_plot.png")
    plt.savefig(f"{output_dir}/combined_psnr_plot.png", dpi=300)
    plt.close()


def visualize_dct_basis(block_size=8, output_dir="output_plots"):
    """
    Visualize the DCT basis functions and save the figure.

    Inputs:
    -----------
    block_size : int
        Size of the DCT blocks.
    output_dir : str
        Directory to save output plots.
    """
    # Create output directory if it doesn't exist
    import os

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))

    # Plot DCT basis functions (for supplemental visualization)
    for i in range(block_size):
        for j in range(block_size):
            n = i * block_size + j
            plt.subplot(block_size, block_size, n + 1)

            basis = np.zeros((block_size, block_size))
            basis[i, j] = 1
            plt.imshow(DCTCompression(block_size=block_size).idct2(basis), cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    plt.suptitle("DCT Basis Functions", fontsize=16)
    plt.subplots_adjust(top=0.95)
    print(f"Saving DCT basis functions plot to {output_dir}/dct_basis_functions.png")
    plt.savefig(f"{output_dir}/dct_basis_functions.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Define image paths using our available sample images
    image_paths = [
        "sample_image/airplane.png",
        "sample_image/arctichare.png",
        "sample_image/baboon.png",
        "sample_image/cat.png",
        "sample_image/boat.png",
        "sample_image/barbara.png",
        "sample_image/goldhill.png",
        "sample_image/monarch.png",
        "sample_image/mountain.png",
        "sample_image/peppers.png",
        "sample_image/fruits.png",
        "sample_image/girl.png",
    ]

    # Create output directory
    output_dir = "dct_compression_results"
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Run DCT experiments
    results = run_dct_experiment(
        image_paths=image_paths, K_values=[2, 4, 8, 16, 32], block_size=8
    )

    # Display and save results
    display_dct_results(image_paths, results, output_dir=output_dir)

    # Visualize and save DCT basis functions
    visualize_dct_basis(block_size=8, output_dir=output_dir)

    # Determine satisfactory reconstruction and save results to text file
    print("\nDetermining satisfactory reconstruction:")

    # Define a PSNR threshold for "satisfactory" reconstruction
    psnr_threshold = (
        30.0  # This threshold can be adjusted based on subjective quality assessment
    )

    # Create a file to save the textual results
    with open(f"{output_dir}/reconstruction_results.txt", "w") as f:
        f.write("Determining satisfactory reconstruction:\n")

        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_results = results[image_name]
            K_values = sorted(image_results.keys())

            f.write(f"\nImage: {image_name}\n")
            f.write(
                f"PSNR threshold for satisfactory reconstruction: {psnr_threshold} dB\n"
            )

            print(f"\nImage: {image_name}")
            print(
                f"PSNR threshold for satisfactory reconstruction: {psnr_threshold} dB"
            )

            for K in K_values:
                psnr = image_results[K]["psnr"]
                status = "Satisfactory" if psnr >= psnr_threshold else "Unsatisfactory"

                f.write(f"  K={K}: PSNR={psnr:.2f} dB - {status}\n")
                print(f"  K={K}: PSNR={psnr:.2f} dB - {status}")

            # Find the minimum K that achieves satisfactory reconstruction
            satisfactory_k = next(
                (k for k in K_values if image_results[k]["psnr"] >= psnr_threshold),
                None,
            )

            if satisfactory_k is not None:
                f.write(
                    f"Minimum K for satisfactory reconstruction: {satisfactory_k}\n"
                )
                print(f"Minimum K for satisfactory reconstruction: {satisfactory_k}")
            else:
                f.write(
                    "No K value achieved satisfactory reconstruction based on the threshold.\n"
                )
                print(
                    "No K value achieved satisfactory reconstruction based on the threshold."
                )
