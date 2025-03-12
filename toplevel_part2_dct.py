from helper_proj2 import *

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

        # Iterate over each image
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

            # Write PSNR values for each K and determine if it is satisfactory
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

            # Save the minimum K value to the file
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
