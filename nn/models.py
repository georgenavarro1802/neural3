from __future__ import unicode_literals

from django.db import models

# Create your models here.


class NeuralFile(models.Model):
    csv = models.FileField(upload_to='csv', blank=True, null=True)

    class Meta:
        verbose_name = 'Neural File'
        verbose_name_plural = 'Neural Files'
        db_table = 'neural_file'

    def file_name(self):
        return self.csv.name.split('/')[1]


class NeuralInputs(models.Model):
    n1 = models.IntegerField(default=0)
    n2 = models.IntegerField(default=0)
    n3 = models.IntegerField(default=0)
    n4 = models.IntegerField(default=0)
    n5 = models.IntegerField(default=0)
    output = models.IntegerField(default=0)

    def __str__(self):
        return "{0}-{1}-{2}-{3}-{4} [{5}]".format(self.n1, self.n2, self.n3, self.n4, self.n5, self.output)

    class Meta:
        verbose_name = 'Neural Input'
        verbose_name_plural = 'Neural Inputs'
        db_table = 'neural_inputs'
