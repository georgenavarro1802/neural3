# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2016-11-12 21:00
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nn', '0002_auto_20161112_1253'),
    ]

    operations = [
        migrations.CreateModel(
            name='NeuralInputs',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('n1', models.IntegerField(default=0)),
                ('n2', models.IntegerField(default=0)),
                ('n3', models.IntegerField(default=0)),
                ('n4', models.IntegerField(default=0)),
                ('n5', models.IntegerField(default=0)),
                ('output', models.IntegerField(default=0)),
            ],
            options={
                'db_table': 'neural_inputs',
                'verbose_name': 'Neural Input',
                'verbose_name_plural': 'Neural Inputs',
            },
        ),
    ]