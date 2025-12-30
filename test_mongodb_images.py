"""
Test script to verify MongoDB image storage implementation.
This script tests that images are correctly stored in MongoDB and can be retrieved.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.data.database import db
from app.data.managers.analysis_manager import analysis_manager
from app.data.models.analysis import (
    AnalysisCaseCreate, ImageData, ClinicalData, ModelOutputs, ModelOutput,
    ShapAnalysis, ShapFeature, TemporalData
)
from datetime import datetime
import uuid


async def test_image_storage():
    """Test that images can be stored and retrieved from MongoDB."""

    print("=" * 60)
    print("Testing MongoDB Image Storage")
    print("=" * 60)

    try:
        # Step 1: Connect to MongoDB
        print("\n[1/5] Connecting to MongoDB...")
        await db.connect()
        print("[OK] Connected to MongoDB")

        # Step 2: Create fake image data
        print("\n[2/5] Creating test image data...")
        test_image_bytes = b"FAKE_IMAGE_DATA_FOR_TESTING_" + b"X" * 1000  # 1KB fake image
        analysis_id = f"AN-TEST-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Create test analysis case with image data
        test_analysis = AnalysisCaseCreate(
            analysis_id=analysis_id,
            patient_id="PAT-TEST",
            lesion_id="LES-TEST",
            image=ImageData(
                filename=f"{analysis_id}.jpg",
                path=f"mongodb://{analysis_id}",
                content_type="image/jpeg",
                data=test_image_bytes  # Binary image data
            ),
            clinical_data=ClinicalData(
                age_at_capture=45,
                lesion_size_mm=6.5
            ),
            model_outputs=ModelOutputs(
                image_only_model=ModelOutput(malignant_probability=0.72),
                clinical_ml_model=ModelOutput(malignant_probability=0.58),
                extracted_features=[
                    {"feature_name": "test_feature", "value": 1.0}
                ]
            ),
            shap_analysis=ShapAnalysis(
                prediction=0.58,
                base_value=0.42,
                features=[
                    ShapFeature(
                        feature="test_feature",
                        display_name="Test Feature",
                        value=1.0,
                        shap_value=0.5,
                        impact="increases"
                    )
                ]
            ),
            temporal_data=TemporalData(
                capture_date=datetime.utcnow(),
                days_since_first_observation=0
            )
        )

        print("[OK] Test data created")
        print(f"   Analysis ID: {analysis_id}")
        print(f"   Image size: {len(test_image_bytes)} bytes")

        # Step 3: Save to MongoDB
        print("\n[3/5] Saving to MongoDB...")
        object_id = await analysis_manager.create_analysis(test_analysis)
        print("[OK] Analysis saved to MongoDB")
        print(f"   Object ID: {object_id}")

        # Step 4: Retrieve from MongoDB
        print("\n[4/5] Retrieving from MongoDB...")
        retrieved = await analysis_manager.get_analysis(analysis_id)

        if not retrieved:
            print("[FAIL] Failed to retrieve analysis")
            return False

        print("[OK] Analysis retrieved successfully")

        # Step 5: Verify image data
        print("\n[5/5] Verifying image data...")

        if not retrieved.image.data:
            print("[FAIL] Image data is None or empty")
            return False

        if retrieved.image.data != test_image_bytes:
            print("[FAIL] Image data mismatch!")
            print(f"   Expected: {len(test_image_bytes)} bytes")
            print(f"   Got: {len(retrieved.image.data)} bytes")
            return False

        print("[OK] Image data verified")
        print(f"   Filename: {retrieved.image.filename}")
        print(f"   Path: {retrieved.image.path}")
        print(f"   Content-Type: {retrieved.image.content_type}")
        print(f"   Size: {len(retrieved.image.data)} bytes")
        print("   Data matches: YES")

        # Clean up: Delete test analysis
        print("\n[Cleanup] Deleting test analysis...")
        deleted = await analysis_manager.delete_analysis(analysis_id)
        if deleted:
            print("[OK] Test analysis deleted")

        print("\n" + "=" * 60)
        print("SUCCESS: All tests passed!")
        print("=" * 60)
        print("\nConclusion:")
        print("[OK] Images can be stored in MongoDB")
        print("[OK] Images can be retrieved from MongoDB")
        print("[OK] Binary data is preserved correctly")
        print("[OK] Ready for Render deployment")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Disconnect from MongoDB
        print("\nDisconnecting from MongoDB...")
        await db.disconnect()
        print("[OK] Disconnected")


async def main():
    """Main test function."""
    success = await test_image_storage()

    if success:
        print("\n[OK] MongoDB image storage is working correctly!")
        sys.exit(0)
    else:
        print("\n[FAIL] MongoDB image storage test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
