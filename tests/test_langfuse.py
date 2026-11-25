
from langfuse import get_client
from backend.config import validate_config

def test_langfuse_connection():
    validate_config()

    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="stage-1-connection-test"
    ) as span:

        span.update(
            input="Testing AI Evaluation Platform",
            output="Langfuse integration successful"
        )

    langfuse.flush()

    print("\nLANGFUSE CONNECTION TEST COMPLETED")
    print("Test observation submitted.")


if __name__ == "__main__":
    test_langfuse_connection()
