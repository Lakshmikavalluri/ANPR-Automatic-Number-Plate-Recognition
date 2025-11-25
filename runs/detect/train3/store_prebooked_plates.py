import os

def load_prebooked_plates(file_path):
    """Load prebooked plates from the text file."""
    try:
        with open(file_path, "r") as file:
            plates = file.readlines()
        # Strip newline characters from each plate
        return [plate.strip() for plate in plates]
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found. Creating a new one.")
        return []

def save_prebooked_plates(plates, file_path):
    """Save the prebooked plates to the text file."""
    with open(file_path, "w") as file:
        for plate in plates:
            file.write(plate + "\n")
    print(f"✅ Prebooked plates saved to {file_path}")

def add_plate(plates, plate):
    """Add a new plate to the list."""
    if plate.upper() not in plates:
        plates.append(plate.upper())  # Ensure plate is in uppercase
        print(f"✅ Plate {plate} added to the prebooked list.")
    else:
        print(f"🚫 Plate {plate} is already in the prebooked list.")

def display_plates(plates):
    """Display the prebooked plates."""
    print("\n=== Prebooked Plates ===")
    if plates:
        for plate in plates:
            print(f"- {plate}")
    else:
        print("No prebooked plates found.")

def generate_prebooked_from_dataset(dataset_path):
    """Generate a list of prebooked plates from a trained dataset."""
    plates = []
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as file:
            lines = file.readlines()
            # Assuming each line in the dataset has the plate number
            plates = [line.strip() for line in lines]
        print(f"✅ Generated prebooked plates from dataset: {len(plates)} plates found.")
    else:
        print(f"❌ Dataset file '{dataset_path}' not found.")
    return plates

# Main code to interact with the user
if __name__ == "__main__":
    # Ask for file paths for prebooked plates and dataset
    file_path = input("Enter the path for prebooked plates file (e.g., prebooked_plates.txt): ").strip()
    dataset_path = input("Enter the path for your dataset file (e.g., dataset.txt): ").strip()

    prebooked_plates = load_prebooked_plates(file_path)

    while True:
        print("\n=== Prebooked Plates Management ===")
        print("1. View Prebooked Plates")
        print("2. Add a Plate")
        print("3. Generate Prebooked Plates from Dataset")
        print("4. Save and Exit")
        choice = input("Select an option (1/2/3/4): ")

        if choice == "1":
            display_plates(prebooked_plates)
        elif choice == "2":
            plate = input("Enter the plate number to add: ").strip()
            add_plate(prebooked_plates, plate)
        elif choice == "3":
            prebooked_plates_from_dataset = generate_prebooked_from_dataset(dataset_path)
            prebooked_plates.extend(prebooked_plates_from_dataset)  # Add dataset plates to the list
            display_plates(prebooked_plates)
        elif choice == "4":
            save_prebooked_plates(prebooked_plates, file_path)
            break
        else:
            print("Invalid choice, please try again.")
