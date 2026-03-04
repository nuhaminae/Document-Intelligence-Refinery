# src/utils/preprocessor.py

import hashlib
import json
import os
import shutil
from datetime import datetime


class Deduplicator:
    def __init__(
        self, folder_path="data", log_path=".refinery/deduplication_log.jsonl"
    ):
        """
        Initialise the Deduplicator object.

        :param folder_path: Path to the folder containing PDFs
        :param log_path: Path to the deduplication log file
        """
        self.folder_path = folder_path
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def hash_file(self, path, block_size=65536):
        """Calculates the SHA256 hash of a file by reading it in blocks of a specified size."""
        hasher = hashlib.sha256()
        # Open the file in binary mode
        with open(path, "rb") as f:
            # and read it in blocks
            for block in iter(lambda: f.read(block_size), b""):
                # and update the hash with each block
                hasher.update(block)
        # and return the hex digest string
        return hasher.hexdigest()

    def move_duplicates(self, duplicates_folder="duplicates"):
        """Find and move duplicate PDFs into a separate folder."""
        # Dictionary mapping file hashes → original file name
        seen = {}
        # List of duplicate records
        duplicates = []

        # Ensure duplicates folder exists
        dup_path = os.path.join(self.folder_path, duplicates_folder)
        os.makedirs(dup_path, exist_ok=True)

        # Iterate over PDF files
        for file in os.listdir(self.folder_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, file)
                file_hash = self.hash_file(file_path)

                # Check if file hash is already in seen
                if file_hash in seen:
                    # Duplicate found → move to duplicates folder
                    target_path = os.path.join(dup_path, file)
                    shutil.move(file_path, target_path)

                    # Add duplication record
                    duplicates.append(
                        {
                            "duplicate_file": file,
                            "original_file": seen[file_hash],
                            "moved_to": target_path,
                        }
                    )
                # If not seen, add to dictionary as original
                else:
                    seen[file_hash] = file
        # Returns list of duplicate records
        return duplicates

    def run(self):
        """
        Runs the deduplication process, moving duplicates to a separate folder and logging the results to .refinery/deduplication_log.jsonl.

        :return: None
        """
        duplicates = self.move_duplicates()

        # Log results to .refinery/deduplication_log.jsonl
        if duplicates:
            with open(self.log_path, "a", encoding="utf-8") as log_file:
                for entry in duplicates:
                    log_entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "duplicate_file": entry["duplicate_file"],
                        "original_file": entry["original_file"],
                        "moved_to": entry["moved_to"],
                    }
                    log_file.write(json.dumps(log_entry) + "\n")

            print(
                f"Moved {len(duplicates)} duplicate(s). Log written to {self.log_path}"
            )
        else:
            print("No duplicates found.")


# Run the deduplication process
if __name__ == "__main__":
    dedup = Deduplicator(folder_path="data")
    dedup.run()
