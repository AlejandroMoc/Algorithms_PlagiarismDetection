import unittest
import os
from pdf_reader import read_pdf as read_document, read_pdf_metadata as read_document_metadata
from pdf_splitter import split_pdf as divide_pdf, split_pdf_chunks as divide_pdf_chunks
from pdf_merger import merge_pdfs as join_pdfs, merge_pdfs_with_pages as join_pdfs_with_pages
from pdf_manipulator import rotate_pdf_pages as rotate_pages, crop_pdf_pages as crop_pages
from pdf_security import encrypt_pdf as secure_pdf, decrypt_pdf as unsecure_pdf

class TestPDFManipulation(unittest.TestCase):

    def setUp(self):
        self.test_pdf = "test.pdf"  # Path to a test PDF file
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def test_read_document(self):
        result = read_document(self.test_pdf)
        self.assertIn('num_pages', result)
        self.assertIsInstance(result['content'], list)
        self.assertGreater(len(result['content']), 0, "PDF content should not be empty")

    def test_read_document_metadata(self):
        metadata = read_document_metadata(self.test_pdf)
        self.assertIn('/Author', metadata)
        self.assertIsInstance(metadata['/Author'], str)

    def test_divide_pdf(self):
        output_files = divide_pdf(self.test_pdf, self.output_dir)
        self.assertTrue(all(os.path.exists(f) for f in output_files))
        self.assertGreater(len(output_files), 0, "No pages were split from the PDF.")

    def test_divide_pdf_invalid(self):
        invalid_pdf = "non_existent.pdf"
        output_files = divide_pdf(invalid_pdf, self.output_dir)
        self.assertEqual(output_files, [], "Expected no output for non-existent PDF.")

    def test_join_pdfs(self):
        pdf_files = [self.test_pdf, self.test_pdf]  # Using the same PDF for testing
        output_path = os.path.join(self.output_dir, "merged.pdf")
        self.assertTrue(join_pdfs(pdf_files, output_path))
        self.assertTrue(os.path.exists(output_path))

    def test_rotate_pages(self):
        output_path = os.path.join(self.output_dir, "rotated.pdf")
        self.assertTrue(rotate_pages(self.test_pdf, output_path, 90))
        self.assertTrue(os.path.exists(output_path))
        # Optionally, check if rotation has actually changed the content (e.g., visually or through libraries)

    def test_crop_pages(self):
        output_path = os.path.join(self.output_dir, "cropped.pdf")
        self.assertTrue(crop_pages(self.test_pdf, output_path, (10, 10, 200, 200)))
        self.assertTrue(os.path.exists(output_path))

    def test_secure_pdf(self):
        output_path = os.path.join(self.output_dir, "encrypted.pdf")
        self.assertTrue(secure_pdf(self.test_pdf, output_path, "user123", "owner456"))
        self.assertTrue(os.path.exists(output_path))
        # Optionally, verify encryption with a failed decryption test case

    def test_unsecure_pdf(self):
        encrypted_path = os.path.join(self.output_dir, "encrypted.pdf")
        decrypted_path = os.path.join(self.output_dir, "decrypted.pdf")
        self.assertTrue(unsecure_pdf(encrypted_path, decrypted_path, "user123"))
        self.assertTrue(os.path.exists(decrypted_path))

    def test_unsecure_invalid_pdf(self):
        invalid_pdf = "non_encrypted.pdf"
        decrypted_path = os.path.join(self.output_dir, "decrypted_invalid.pdf")
        self.assertFalse(unsecure_pdf(invalid_pdf, decrypted_path, "user123"))
        self.assertFalse(os.path.exists(decrypted_path), "Decryption should fail for non-encrypted PDF.")

    def tearDown(self):
        # Clean up output directory
        for file_name in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file_name))
        os.rmdir(self.output_dir)

if __name__ == '__main__':
    unittest.main()