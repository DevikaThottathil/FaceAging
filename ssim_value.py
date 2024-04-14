
# Define SSIM calculation function
def ssim(img1, img2, window_size=11, size_average=True):
    # Based on Structural Similarity Index implementation in PyTorch
    # https://pytorch.org/docs/stable/torchvision/metrics.html#torchvision.metrics.ssim
    window = create_window(window_size).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=3)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=3) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window(window_size, channel=3):
    sigma = 1.5
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    gauss = gauss.unsqueeze(0).unsqueeze(0)  # Adjust tensor dimensions
    return gauss.expand(channel, 1, window_size, window_size).contiguous()

# View function for user's homepage
def userhome(request):
    if request.method == 'POST' and request.FILES.get('fileUpload'):
        uploaded_file = request.FILES['fileUpload']
        fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'app/static/media'))
        filename = fs.save(uploaded_file.name, uploaded_file)
        media = settings.MEDIA_ROOT
        img_path = os.path.join(media, filename)
        
        # Open and preprocess the uploaded image
        img = Image.open(img_path).convert('RGB')
        img_tensor = trans(img).unsqueeze(0)
        
        # Generate aged face using the model
        aged_face = model(img_tensor)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0

        # Save the generated image
        output_path = os.path.join(media, 'output.png')
        aged_face_pil = Image.fromarray((aged_face * 255).astype('uint8'))
        aged_face_pil.save(output_path)

        # Calculate SSIM between original and generated images
        original_img_tensor = img_tensor / 2.0 + 0.5  # Denormalize
        original_img_tensor = original_img_tensor.to('cpu')  # Move to CPU
        generated_img_tensor = torch.from_numpy(aged_face.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        generated_img_tensor = generated_img_tensor.to('cpu')  # Move to CPU
        ssim_value = ssim(original_img_tensor, generated_img_tensor)

        # Prepare context for rendering result.html
        context = {
            'key': fs.url(output_path),
            'key2': fs.url(filename),
            'ssim_value': ssim_value.item()
        }
        return render(request, 'result.html', context)
    return render(request, 'user_home.html')
