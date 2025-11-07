"""
🎯 FACE SWAP WITH YOUR OWN IMAGES - SIMPLE EXAMPLE
=================================================

This is the easiest way to swap faces using your own two images!

STEP-BY-STEP INSTRUCTIONS:
1. Put your images in a folder called 'my_photos'
2. Name them something like 'person1.jpg' and 'person2.jpg'  
3. Update the paths below to match your image names
4. Run this script!

The AI will swap the face from person1 onto person2's body/background.
"""

from main.face_swapper import swap_faces
import os


def simple_face_swap():
    """
    🚀 THE SIMPLEST WAY TO SWAP FACES WITH YOUR IMAGES
    =================================================
    
    Just update the image paths below and run this function!
    """
    
    print("🎯 SIMPLE FACE SWAP EXAMPLE")
    print("=" * 40)
    
    # 🔧 CHANGE THESE TO YOUR IMAGE PATHS! 🔧
    # ========================================
    
    # Image 1: The person whose FACE you want to put on someone else
    source_face = "my_photos/person1.jpg"
    
    # Image 2: The person who will RECEIVE the new face (their body/background stays)
    target_person = "my_photos/person2.jpg"
    
    # Where to save the result
    output_file = "my_face_swap_result.png"
    
    # ========================================
    
    print(f"📸 Taking face from: {source_face}")
    print(f"📸 Putting it on: {target_person}")
    print(f"💾 Saving result to: {output_file}")
    
    # Check if images exist
    if not os.path.exists(source_face):
        print(f"❌ Source image not found: {source_face}")
        print("💡 Solution: Create 'my_photos' folder and put your images there!")
        return
        
    if not os.path.exists(target_person):
        print(f"❌ Target image not found: {target_person}")
        print("💡 Solution: Create 'my_photos' folder and put your images there!")
        return
    
    print("\n🚀 Starting face swap...")
    
    try:
        # This is where the magic happens!
        result = swap_faces(
            source_image_path=source_face,      # Face to be swapped
            target_image_path=target_person,    # Person receiving the face
            checkpoint_path='checkpoints/ddpm_faceswap_checkpoint',  # Trained model
            num_denoising_steps=50,             # Quality vs speed (higher = better quality)
            save_result=output_file             # Where to save
        )
        
        print(f"✅ SUCCESS! Face swap completed!")
        print(f"🎉 Check your result: {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("💡 You need to train the model first!")
        print("   Set mode='train' in main.py and run it first.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your images are clear photos with visible faces!")


def setup_example_folder():
    """
    🛠️ HELPER FUNCTION - Creates the folder structure you need
    """
    print("🛠️ Setting up example folder structure...")
    
    # Create my_photos folder if it doesn't exist
    if not os.path.exists("my_photos"):
        os.makedirs("my_photos")
        print("📁 Created 'my_photos' folder")
    else:
        print("📁 'my_photos' folder already exists")
    
    print("\n📋 TO USE THIS SCRIPT:")
    print("1. Put your photos in the 'my_photos' folder")
    print("2. Name them 'person1.jpg' and 'person2.jpg' (or update the paths above)")
    print("3. Run simple_face_swap()")
    print("\n💡 Tips for best results:")
    print("   - Use clear, front-facing photos")
    print("   - Make sure faces are well-lit and visible")
    print("   - Photos should be reasonably high quality")


def multiple_face_swaps():
    """
    🔄 ADVANCED EXAMPLE - Swap multiple pairs of faces at once
    """
    print("🔄 MULTIPLE FACE SWAPS")
    print("=" * 30)
    
    # Define multiple image pairs to swap
    image_pairs = [
        ("my_photos/person1.jpg", "my_photos/person2.jpg"),
        ("my_photos/person3.jpg", "my_photos/person4.jpg"),
        # Add more pairs here...
    ]
    
    print(f"Processing {len(image_pairs)} face swap pairs...")
    
    for i, (source, target) in enumerate(image_pairs):
        print(f"\n--- Pair {i+1}: {source} → {target} ---")
        
        if os.path.exists(source) and os.path.exists(target):
            try:
                result = swap_faces(
                    source_image_path=source,
                    target_image_path=target,
                    save_result=f"face_swap_result_{i+1}.png"
                )
                print(f"✅ Pair {i+1} completed!")
            except Exception as e:
                print(f"❌ Pair {i+1} failed: {e}")
        else:
            print(f"❌ Images not found for pair {i+1}")
    
    print("\n🎉 Multiple face swaps completed!")


if __name__ == "__main__":
    """
    🚀 RUN THIS SCRIPT DIRECTLY TO TEST FACE SWAPPING!
    """
    print("🎯 FACE SWAP SCRIPT STARTED")
    print("=" * 50)
    
    # Set up the folder structure
    setup_example_folder()
    
    print("\n" + "=" * 50)
    
    # Run the simple face swap
    simple_face_swap()
    
    # Uncomment the line below if you want to test multiple face swaps
    # multiple_face_swaps()