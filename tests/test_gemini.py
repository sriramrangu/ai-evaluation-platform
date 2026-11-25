
import time

from google import genai
from google.genai import errors

from backend.config import GEMINI_API_KEY, GEMINI_MODEL


def test_gemini_connection():

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, 4):

        try:
            print(f"\nAttempt {attempt}: Connecting to Gemini...")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Explain AI evaluation in two sentences."
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response")

            print("\nGEMINI CONNECTION SUCCESSFUL")
            print("\nGenerated Response:")
            print(response.text)

            return

        except errors.APIError as e:

            if e.code in (429, 500, 502, 503, 504):
                if attempt == 3:
                    print("\nGemini service is currently unavailable.")
                    print("Please retry after some time.")
                    return

                delay = 5 * (2 ** (attempt - 1))

                print(f"Temporary API error: {e.code}")
                print(f"Retrying in {delay} seconds...")

                time.sleep(delay)

            else:
                raise


if __name__ == "__main__":
    test_gemini_connection()
