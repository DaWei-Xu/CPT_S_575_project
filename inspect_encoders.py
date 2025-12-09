import os
import pickle
import numpy as np

ENCODER_DIR = "data/processed_data/label_encoders"

def show_label_mapping(col_name):
    """
    col_name 例如：'genus (Reference)' / 'family (Reference)'
    """
    fname = col_name.replace(" ", "_") + "_encoder.pkl"
    path = os.path.join(ENCODER_DIR, fname)

    print("\n=== ", col_name, " ===")
    print("Loading:", path)
    with open(path, "rb") as f:
        le = pickle.load(f)

    classes = le.classes_
    for idx, cls in enumerate(classes):
        print(f"{idx:4d} -> {repr(cls)}")

    # 专门看看 Unassigned / Unknown
    lower = np.char.lower(classes.astype(str))
    mask = np.isin(lower, ["unassigned", "unknown", ""])
    print("Possible unassigned codes:")
    for idx in np.where(mask)[0]:
        print(f"  id={idx:4d}, label={repr(classes[idx])}")


if __name__ == "__main__":
    show_label_mapping("realm (Reference)")
    show_label_mapping("phylum (Reference)")
    show_label_mapping("class (Reference)")
    show_label_mapping("order (Reference)")
    show_label_mapping("family (Reference)")
    show_label_mapping("subfamily (Reference)")
    show_label_mapping("genus (Reference)")
