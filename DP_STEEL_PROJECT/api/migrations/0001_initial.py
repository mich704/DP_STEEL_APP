# Generated by Django 3.2.25 on 2024-09-20 13:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AIModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('path', models.CharField(max_length=400, unique=True)),
            ],
            options={
                'db_table': 'ai_models',
            },
        ),
        migrations.CreateModel(
            name='ExtractedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(max_length=400, unique=True)),
            ],
            options={
                'db_table': 'extracted_images',
            },
        ),
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(max_length=400, unique=True)),
                ('label', models.CharField(choices=[('microstructure', 'microstructure'), ('rest', 'rest')], max_length=255, null=True)),
            ],
            options={
                'db_table': 'images',
            },
        ),
        migrations.CreateModel(
            name='Publication',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('author', models.CharField(max_length=255, null=True)),
                ('path', models.CharField(max_length=400, unique=True)),
                ('filename', models.CharField(max_length=300, unique=True)),
                ('type', models.CharField(choices=[('unlabelled', 'unlabelled'), ('labelled', 'labelled')], max_length=255, null=True)),
                ('creation_date_raw', models.CharField(max_length=255, null=True)),
                ('title', models.CharField(max_length=255, null=True)),
                ('keywords', models.CharField(max_length=255, null=True)),
            ],
            options={
                'db_table': 'publications',
            },
        ),
        migrations.CreateModel(
            name='PreprocessedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(max_length=400, unique=True)),
                ('extracted_image_parent', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='api.extractedimage')),
                ('image_parent', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='api.image')),
            ],
            options={
                'db_table': 'preprocessed_images',
            },
        ),
        migrations.AddField(
            model_name='image',
            name='publication',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='api.publication'),
        ),
        migrations.AddField(
            model_name='extractedimage',
            name='publication',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.publication'),
        ),
        migrations.CreateModel(
            name='ClassifiedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(max_length=400, unique=True)),
                ('label', models.CharField(choices=[('microstructure', 'microstructure'), ('rest', 'rest')], max_length=255)),
                ('ai_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.aimodel')),
                ('image_parent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.preprocessedimage')),
            ],
            options={
                'db_table': 'classified_images',
            },
        ),
    ]
