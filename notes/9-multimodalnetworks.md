## Multimodel
Unlike traditional unimodal models that operate in isolation (e.g., a text-only LLM or an image-only CNN), a multimodal model maps different types of inputs into a shared mathematical space. This allows the network to natively understand the relationship between a spoken word, a written sentence, and a visual object.

## Text-to-Image Generation models
- Architecture components: Language understanding (converting text to mathematical representations), image generation (from noise using diffusion), inpainting (selectively edit parts of image), outpainting (expand images beyond original boundaries)
- Prompt: The quality of the image generated depends on text prompt and other model parameters. The prompt should have all precise details like the building name, ivy, lighting, photography etc. 
- Parameters and control: seed (to enable stable outputs and reproducibility), steps (no of diffusion steps), guidance scale (creative vs strict prompt adherence)
- API Access: Ensure proper error handling (rate limits, network issues, invalid prompt) and response handling (response metadata for tracking, debugging and improving prompts)
- Ethical considerations: Copyright and Ownership, Bias and Representation, Impact on creative industries, 
- Key guidelines for responsible AI: Transparent usage, Attribution and compensation, Quality control, regular review

