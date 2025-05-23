import unittest
import os
from pdf_processor import process_pdf, process_pdf_metadata
from pdf_divider import divide_pdf, divide_pdf_parts
from pdf_combiner import combine_pdfs, combine_pdfs_with_parts
from pdf_modifier import modify_pdf_pages, adjust_pdf_pages
from pdf_protector import secure_pdf, unsecure_pdf

class TestPDFHandling(unittest.TestCase):

    def setUp(self):
        self.sample_pdf = "sample.pdf"  # Path to a sample PDF file
        self.output_folder = "result"
        os.makedirs(self.output_folder, exist_ok=True)

    def test_process_pdf(self):
        result = process_pdf(self.sample_pdf)
        self.assertIn('total_pages', result)
        self.assertIsInstance(result['contents'], list)
        self.assertGreater(len(result['contents']), 0, "PDF contents should not be empty")

    def test_process_pdf_metadata(self):
        metadata = process_pdf_metadata(self.sample_pdf)
        self.assertIn('/Creator', metadata)
        self.assertIsInstance(metadata['/Creator'], str)

    def test_divide_pdf(self):
        output_files = divide_pdf(self.sample_pdf, self.output_folder)
        self.assertTrue(all(os.path.exists(f) for f in output_files))
        self.assertGreater(len(output_files), 0, "No pages were divided from the PDF.")

    def test_divide_pdf_invalid(self):
        invalid_pdf = "nonexistent.pdf"
        output_files = divide_pdf(invalid_pdf, self.output_folder)
        self.assertEqual(output_files, [], "Expected no output for non-existent PDF.")

    def test_combine_pdfs(self):
        pdf_files = [self.sample_pdf, self.sample_pdf]  # Using the same PDF for testing
        output_path = os.path.join(self.output_folder, "merged.pdf")
        self.assertTrue(combine_pdfs(pdf_files, output_path))
        self.assertTrue(os.path.exists(output_path))

    def test_modify_pdf_pages(self):
        output_path = os.path.join(self.output_folder, "modified.pdf")
        self.assertTrue(modify_pdf_pages(self.sample_pdf, output_path, 90))
        self.assertTrue(os.path.exists(output_path))
        # Optionally, check if modification has actually changed the content (e.g., visually or through libraries)

    def test_adjust_pdf_pages(self):
        output_path = os.path.join(self.output_folder, "adjusted.pdf")
        self.assertTrue(adjust_pdf_pages(self.sample_pdf, output_path, (10, 10, 200, 200)))
        self.assertTrue(os.path.exists(output_path))

    def test_secure_pdf(self):
        output_path = os.path.join(self.output_folder, "protected.pdf")
        self.assertTrue(secure_pdf(self.sample_pdf, output_path, "user123", "owner456"))
        self.assertTrue(os.path.exists(output_path))
        # Optionally, verify protection with a failed unprotection test case

    def test_unsecure_pdf(self):
        protected_path = os.path.join(self.output_folder, "protected.pdf")
        unprotected_path = os.path.join(self.output_folder, "unprotected.pdf")
        self.assertTrue(unsecure_pdf(protected_path, unprotected_path, "user123"))
        self.assertTrue(os.path.exists(unprotected_path))

    def test_unsecure_invalid_pdf(self):
        invalid_pdf = "nonprotected.pdf"
        unprotected_path = os.path.join(self.output_folder, "unprotected_invalid.pdf")
        self.assertFalse(unsecure_pdf(invalid_pdf, unprotected_path, "user123"))
        self.assertFalse(os.path.exists(unprotected_path), "Unprotection should fail for non-protected PDF.")

    def tearDown(self):
        # Clean up output directory
        for file in os.listdir(self.output_folder):
            os.remove(os.path.join(self.output_folder, file))
        os.rmdir(self.output_folder)

if __name__ == '__main__':
    unittest.main()