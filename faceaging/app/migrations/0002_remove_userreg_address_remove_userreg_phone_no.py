# Generated by Django 5.0.3 on 2024-03-16 11:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userreg',
            name='address',
        ),
        migrations.RemoveField(
            model_name='userreg',
            name='phone_no',
        ),
    ]
