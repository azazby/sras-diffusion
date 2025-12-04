import os
import torch
from tqdm import tqdm
from PIL import Image
from pytorch_fid import fid_score
import clip

'''
FID score: https://codefinity.com/courses/v2/37287944-3858-4d25-b5b4-8963c453781b/d5c11d4a-e709-4f89-8d63-e34876cc8c8d/e1716d04-ee08-4c84-8604-780f3be53e5d
CLIP Score by OpenAI: https://github.com/openai/CLIP
'''

class ModelEvaluation:
    def __init__(self, batch_size, dims, device='cuda'):
        self.batch_size = batch_size
        self.dims = dims
        self.device = device

        # The model used to compute CLIP scores
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        

    def compute_fid(self, real_img_path, gen_img_path):
        '''
        Compute FID between the original images (in real_img_path) and a set of 
        generated images from an experimental run (in gen_img_path).
        '''
        fid_value = fid_score.calculate_fid_given_paths(
            [real_img_path, gen_img_path], 
            self.batch_size, 
            self.device, 
            self.dims
        )
        return fid_value
    
    
    def compute_clip_score(self, gen_img_path, captions_path):
        '''
        Compute CLIP Score for generated images.
        Args:
        - gen_img_path: Path to directory that contains images generated from 
        an experiment.
        - captions_path: .txt file that contains text prompts for the images in 
        gen_img_path, in order.
        '''
        with open(captions_path, 'r') as f:
            captions = [line.strip() for line in f.readlines()]
        
        image_files = sorted([
            os.path.join(gen_img_path, f) 
            for f in os.listdir(gen_img_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        assert len(image_files) == len(captions), \
            f"Mismatch: {len(image_files)} images vs {len(captions)} captions"
        
        clip_scores = []
        
        with torch.no_grad():
            for img_path, caption in tqdm(zip(image_files, captions), desc="Computing CLIP scores"):
                image = self.clip_preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(self.device)
                text = clip.tokenize([caption], truncate=True).to(self.device)
                
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Cosine similarity
                similarity = (image_features * text_features).sum(dim=1)
                clip_scores.append(similarity.item() * 100)
        
        return sum(clip_scores) / len(clip_scores)
    
    
    def evaluate(self, real_img_path, gen_img_path, captions_path):
        '''
        Compute both FID and CLIP scores.
        '''
        fid = self.compute_fid(real_img_path, gen_img_path)
        clip_score = self.compute_clip_score(gen_img_path, captions_path)

        results = {
            "FID": fid,
            "CLIP": clip_score
        }

        return results
    

if __name__ == "__main__":
    BATCHSIZE = 50
    IMGDIMS = 2048   # Standard size, don't change this
    
    evaluator = ModelEvaluation(BATCHSIZE, IMGDIMS, device='cuda')
    
    results = evaluator.evaluate(
        real_img_path='path/to/coco/val2017',
        gen_img_path='outputs/baseline',
        captions_path='eval_prompts.txt'
    )
    
    print(f"\nResults:")
    print(f"FID: {results['FID']:.2f}")
    print(f"CLIP Score: {results['CLIP']:.2f}")