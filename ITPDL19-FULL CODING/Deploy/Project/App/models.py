from django.db import models
from django.contrib.auth.models import User
from PIL import Image



# Extending User Model Using a One-To-One Link
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    avatar = models.ImageField(default='default.jpg', upload_to='profile_images')
    bio = models.TextField()

    def __str__(self):
        return self.user.username

    # resizing images
    def save(self, *args, **kwargs):
        super().save()

        img = Image.open(self.avatar.path)

        if img.height > 100 or img.width > 100:
            new_img = (100, 100)
            img.thumbnail(new_img)
            img.save(self.avatar.path)

    
class UserImageModel(models.Model):
    image = models.ImageField(upload_to = 'images/',blank=True)
    label = models.CharField(max_length=20,default='data')
    def __str__(self):
        return str(self.image)
    
class DiabetesAssessment(models.Model):

    Age = models.IntegerField()
    Gender = models.IntegerField()  # Adjust the max_length if needed
    Polyuria = models.CharField(max_length=10)  # Adjust the max_length if needed
    Polydipsia = models.CharField(max_length=10)  # Adjust the max_length if needed
    sudden_weight_loss = models.CharField(max_length=10)  # Adjust the max_length if needed
    weakness = models.CharField(max_length=10)  # Adjust the max_length if needed
    Polyphagia = models.CharField(max_length=10)  # Adjust the max_length if needed
    Genital_thrush = models.CharField(max_length=10)  # Adjust the max_length if needed
    visual_blurring = models.CharField(max_length=10)  # Adjust the max_length if needed
    Itching = models.CharField(max_length=10)  # Adjust the max_length if needed
    Irritability = models.CharField(max_length=10)  # Adjust the max_length if needed
    delayed_healing = models.CharField(max_length=10)  # Adjust the max_length if needed
    partial_paresis = models.CharField(max_length=10)  # Adjust the max_length if needed
    muscle_stiffness = models.CharField(max_length=10)  # Adjust the max_length if needed
    Alopecia = models.CharField(max_length=10)  # Adjust the max_length if needed
    Obesity = models.CharField(max_length=10)
    label = models.CharField(max_length=100, blank=True, null=True)  # This output

    def __str__(self):
        return f'{self.label}'