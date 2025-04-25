import numpy as np
import unittest

class TestBackgroundClassRemoval(unittest.TestCase):
    
    def test_background_class_filtering(self):
        """Test that background class is properly filtered out in postprocessing."""
        
        # Define the sigmoid_stable function (copied from original)
        def sigmoid_stable(x):
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        
        # Define a simplified version of the postprocess function focusing on background class removal
        def postprocess_simplified(bboxes, logits, img_dims, class_names, 
                                  confidence=0.5, max_detections=100, 
                                  resize_method="Stretch to", img_size_h=640, img_size_w=640):
            """Simplified version of postprocess function for testing."""
            
            batch_size, num_queries, num_classes = logits.shape
            logits_sigmoid = sigmoid_stable(logits)

            processed_predictions = []

            # Find background class index if it exists
            background_class_index = -1  # Default to -1 (won't match valid indices)
            background_class_name = "background_class83422"
            try:
                background_class_index = class_names.index(background_class_name)
            except ValueError:
                print(f"Background class {background_class_name} not found in class_names")
                pass

            for batch_idx in range(batch_size):
                orig_h, orig_w = img_dims[batch_idx]

                logits_flat = logits_sigmoid[batch_idx].reshape(-1)

                # Get top-k scores
                sorted_indices = np.argsort(-logits_flat)[:max_detections]
                topk_scores = logits_flat[sorted_indices]

                # Filter by confidence threshold
                conf_mask = topk_scores > confidence
                sorted_indices = sorted_indices[conf_mask]
                topk_scores = topk_scores[conf_mask]

                # Get box indices and class labels
                topk_boxes = sorted_indices // num_classes
                topk_labels = sorted_indices % num_classes

                # Filter background class (THIS IS THE KEY PART WE'RE TESTING)
                if background_class_index != -1:
                    class_filter_mask = topk_labels != background_class_index
                    topk_scores = topk_scores[class_filter_mask]
                    topk_labels = topk_labels[class_filter_mask]
                    topk_boxes = topk_boxes[class_filter_mask]

                # Get selected boxes
                selected_boxes = bboxes[batch_idx, topk_boxes]

                # Simplified box processing
                # Here we skip most of the box transformation logic from the original
                # and just create a dummy array with the right shape
                boxes_xyxy = np.ones((len(topk_boxes), 4)) * 0.5
                
                # Build final predictions array
                batch_predictions = np.column_stack(
                    (
                        boxes_xyxy,  # [x1, y1, x2, y2]
                        topk_scores,  # confidence
                        np.zeros((len(topk_scores), 1), dtype=np.float32),  # placeholder
                        topk_labels,  # class indices
                    )
                )

                processed_predictions.append(batch_predictions)

            return processed_predictions
        
        # ------------------------
        # TEST CASE 1: With background class present in class_names
        # ------------------------
        class_names = ["class1", "background_class83422", "class3"]
        
        # Create synthetic test data
        batch_size = 1
        num_queries = 4
        num_classes = 3  # Matches the length of class_names
        
        # Create bounding boxes
        bboxes = np.ones((batch_size, num_queries, 4), dtype=np.float32) * 0.5
        
        # Create logits - assign high values to make class assignments clear
        logits = np.zeros((batch_size, num_queries, num_classes), dtype=np.float32)
        
        # Box 0 - class 0
        logits[0, 0, 0] = 10.0
        # Box 1 - background class (class 1)
        logits[0, 1, 1] = 10.0
        # Box 2 - class 2
        logits[0, 2, 2] = 10.0
        # Box 3 - background class (class 1) again
        logits[0, 3, 1] = 10.0
        
        # Process predictions
        img_dims = [(480, 640)]  # Original height, width
        processed_preds = postprocess_simplified(
            bboxes, logits, img_dims, class_names, 
            confidence=0.5, max_detections=10
        )
        
        # Check results
        # Should have 2 predictions (boxes 0 and 2), with background class filtered out
        self.assertEqual(processed_preds[0].shape[0], 2, 
                         "Should have 2 predictions with background class filtered out")
        
        # Check class indices (should be 0 and 2, not 1)
        class_indices = processed_preds[0][:, -1]
        self.assertIn(0.0, class_indices, "Class index 0 should be present")
        self.assertIn(2.0, class_indices, "Class index 2 should be present")
        self.assertNotIn(1.0, class_indices, "Background class index should be filtered out")
        
        # ------------------------
        # TEST CASE 2: Without background class in class_names
        # ------------------------
        class_names = ["class1", "class2", "class3"]  # No background class
        
        # Create new synthetic data
        batch_size = 1
        num_queries = 3
        num_classes = 3
        
        bboxes = np.ones((batch_size, num_queries, 4), dtype=np.float32) * 0.5
        logits = np.zeros((batch_size, num_queries, num_classes), dtype=np.float32)
        
        # Each box has a different class
        for i in range(num_queries):
            logits[0, i, i] = 10.0
        
        # Process predictions
        processed_preds = postprocess_simplified(
            bboxes, logits, img_dims, class_names, 
            confidence=0.5, max_detections=10
        )
        
        # Check results
        # Should keep all 3 predictions since there's no background class to filter
        self.assertEqual(processed_preds[0].shape[0], 3, 
                         "Should keep all predictions when no background class exists")
        
        # All class indices should be present
        class_indices = processed_preds[0][:, -1]
        self.assertIn(0.0, class_indices, "Class index 0 should be present")
        self.assertIn(1.0, class_indices, "Class index 1 should be present")
        self.assertIn(2.0, class_indices, "Class index 2 should be present")

if __name__ == "__main__":
    unittest.main()