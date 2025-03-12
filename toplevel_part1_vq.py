from helper_proj2 import *

if __name__ == "__main__":
    import argparse

    # Create command line arguments
    parser = argparse.ArgumentParser(
        description="Vector Quantization using GLA"
    )
    parser.add_argument(
        "--test_images",
        nargs="+",
        default=["sample_image/barbara.png", "sample_image/boat.png"],
        help="Paths to test images",
    )
    parser.add_argument(
        "--training_images",
        nargs="+",
        default=[
            "sample_image/airplane.png",
            "sample_image/arctichare.png",
            "sample_image/baboon.png",
            "sample_image/cat.png",
            "sample_image/fruits.png",
            "sample_image/girl.png",
            "sample_image/goldhill.png",
            "sample_image/monarch.png",
            "sample_image/mountain.png",
            "sample_image/peppers.png",
        ],
        help="Paths to training images for the collection experiment",
    )
    parser.add_argument(
        "--codebook_sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024],
        help="Codebook sizes to test",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4,
        help="Size of the blocks for vector quantization",
    )
    parser.add_argument(
        "--save_codebooks", action="store_true", help="Save trained codebooks to files"
    )

    args = parser.parse_args()

    print("Using our Generalized Lloyd Algorithm (GLA)")

    # Load images
    test_images = load_images(args.test_images)
    training_images = load_images(args.training_images)

    # First experiment: Train and test on the same images
    print("Experiment 1: Training and testing on the same images")

    results_same = {}

    # Train and test for each codebook size
    for codebook_size in args.codebook_sizes:
        vq = VectorQuantization(
            block_size=args.block_size,
        )
        vq.train_codebook(test_images, codebook_size)

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

        results_same[codebook_size] = test_results

    display_results(test_images, results_same, "Same-trained ")

    # Second experiment: Train on a collection of images and test on test images
    print("Experiment 2: Training on a collection of images and testing on test images")

    results_different = {}
    # Train and test for each codebook size
    for codebook_size in args.codebook_sizes:
        vq = VectorQuantization(
            block_size=args.block_size,
        )
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

        results_different[codebook_size] = test_results

    display_results(test_images, results_different, "Collection-trained ")

    # Compare results
    print("\nComparison of results:")
    # Print results for each image
    for codebook_size in args.codebook_sizes:
        print(f"\nCodebook size: {codebook_size}")
        # Print results for each image
        for i in range(len(test_images)):
            same_result = results_same[codebook_size][i]
            diff_result = results_different[codebook_size][i]
            print(f"Image {i+1} ({os.path.basename(args.test_images[i])}):")
            print(
                f"  Same-trained: MSE={same_result['mse']:.2f}, PSNR={same_result['psnr']:.2f} dB, CR={same_result['compression_ratio']:.2f}x"
            )
            print(
                f"  Collection-trained: MSE={diff_result['mse']:.2f}, PSNR={diff_result['psnr']:.2f} dB, CR={diff_result['compression_ratio']:.2f}x"
            )

    # Save codebooks if requested
    if args.save_codebooks:
        algo_name = "gla"

        # Save codebooks from same-image training
        for codebook_size in args.codebook_sizes:
            vq = VectorQuantization(block_size=args.block_size)
            vq.train_codebook(test_images, codebook_size)
            vq.save_codebook(f"codebook_same_{algo_name}_{codebook_size}.npy")
            print(f"Saved codebook_same_{algo_name}_{codebook_size}.npy")

        # Save codebooks from collection training
        for codebook_size in args.codebook_sizes:
            vq = VectorQuantization(block_size=args.block_size)
            vq.train_codebook(training_images, codebook_size)
            vq.save_codebook(f"codebook_collection_{algo_name}_{codebook_size}.npy")
            print(f"Saved codebook_collection_{algo_name}_{codebook_size}.npy")

    # For GLA, plot distortion history to show convergence
    vq = VectorQuantization(block_size=args.block_size)
    blocks = vq.extract_blocks(test_images[0])
    _, distortion_history = vq.generalized_lloyd_algorithm(
        blocks, codebook_size=args.codebook_sizes[0]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(distortion_history)
    plt.xlabel("Iteration")
    plt.ylabel("Average Distortion (MSE)")
    plt.title(f"GLA Convergence for Codebook Size {args.codebook_sizes[0]}")
    plt.grid(True)
    print(f"Saving gla_convergence_{args.codebook_sizes[0]}.png...")
    plt.savefig(f"gla_convergence_{args.codebook_sizes[0]}.png")
    plt.show()
