from transformers import pipeline
import torch
import onnxruntime
from torchvision import transforms


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    device = 0 if torch.cuda.is_available() else -1

    model = onnxruntime.InferenceSession("mtailor.onnx", device=device)


def to_numpy(self, tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    ort_inputs = {model.get_inputs()[0].name: to_numpy(prompt)}
    model_output = model.run(None, ort_inputs)[0]
    to_tensor = transforms.ToTensor()
    onnx_res = torch.argmax(to_tensor(model_output).unsqueeze_(0))
    return onnx_res
