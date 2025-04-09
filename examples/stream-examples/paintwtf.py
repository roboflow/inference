import cv2

import inference
from inference.core.utils.postprocess import cosine_similarity
from inference.models import Clip

clip = Clip()

prompt = "an ace of spades playing card"
text_embedding = clip.embed_text(prompt)

def render(result, image):
    # get the cosine similarity between the prompt & the image
    similarity = cosine_similarity(result["embeddings"][0], text_embedding[0])

    # scale the result to 0-100 based on heuristic (~the best & worst values I've observed)
    range = (0.15, 0.40)
    similarity = (similarity-range[0])/(range[1]-range[0])
    similarity = max(min(similarity, 1), 0)*100
    
    # print the similarity
    text = f"{similarity:.1f}%"
    cv2.putText(image, text, (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 12, (255, 255, 255), 30)
    cv2.putText(image, text, (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 12, (206, 6, 103), 16)

    # print the prompt
    cv2.putText(image, prompt, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10)
    cv2.putText(image, prompt, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 2, (206, 6, 103), 5)

    # display the image
    cv2.imshow("CLIP", image)
    cv2.waitKey(1)

# start the stream
inference.Stream(
    source="webcam",
    model=clip,

    output_channel_order="BGR",
    use_main_thread=True,
    
    on_prediction=render
)
