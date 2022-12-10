# Generated by Django 4.0.3 on 2022-04-09 06:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mainApp', '0007_order_patment_completed_order_payment_method_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='order',
            name='order_status',
            field=models.CharField(choices=[('Order Received', 'Order Received'), ('Order Completed', 'Order Completed'), ('On the way', 'On the way'), ('Order Processing', 'Order Processing'), ('Order Canceled', 'Order Canceled')], max_length=50),
        ),
        migrations.CreateModel(
            name='ProductImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='products/images/')),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mainApp.product')),
            ],
        ),
    ]
