from inference import get_model
from PIL import Image
import time
model = get_model("fooddescription/61", api_key="zaRavHwbvIXpGerDM3wi")
#image = Image.open("360_F_261982444_jDzDlgClqQDc5DX47Qy4PSayvcn89vQi.jpg")
start = time.time()
print(model.infer("https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-1273516682.jpg", prompt="What is this food?"))
end = time.time()
print(f"Time taken: {end - start}")

