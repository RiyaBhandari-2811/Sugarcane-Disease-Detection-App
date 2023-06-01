package com.example.sugarcanediseasedetection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import android.widget.ProgressBar;
import android.widget.Toast;

import com.example.sugarcanediseasedetection.ml.FinalModel;
import com.github.dhaval2404.imagepicker.ImagePicker;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    static final String USER_IMG_KEY = "UserImgKey";
    static final String DISEASE_NAME_KEY = "DiseaseNameKey";
    static final int REQ_CODE = 100;
    Bitmap bitmap;
    Uri uri;

    int imageSize = 224;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        loadLocale();
        setContentView(R.layout.activity_main);

        Button changelng = findViewById(R.id.changeMyLang);
        changelng.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //show alert dialog to display list of languages
                showChangeLanguageDialog();
            }
        });

    }

    private void showChangeLanguageDialog() {
        final String[] listItems = {"हिंदी", "मराठी", "English"}; //array of languages
        AlertDialog.Builder mBuilder = new AlertDialog.Builder(MainActivity.this);
        mBuilder.setTitle("Choose Language...");

        // The setSingleChoiceItems method sets up a single-choice selection behavior for the list of language options.
        mBuilder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                // The setLocale() method is called with the selected language code.
                //The recreate() method is called to recreate the activity with the updated language.
                if (i == 0) {
                    setLocale("hi");
                    recreate();
                }
                if (i == 1) {
                    setLocale("mr");
                    recreate();
                }
                if (i == 2) {
                    setLocale("en");
                    recreate();
                }
                //dismiss alert box
                dialogInterface.dismiss();
            }
        });

        AlertDialog mDialog = mBuilder.create();
        // Shows our dialog box
        mDialog.show();
    }

    private void setLocale(String lang) {
        Locale locale = new Locale(lang);
        Locale.setDefault(locale);
        // A new Configuration object is created, and its locale is set to the created Locale.
        //The application's resources are updated with the new configuration using updateConfiguration().
        Configuration config = new Configuration();
        config.locale = locale;
        getBaseContext().getResources().updateConfiguration(config, getBaseContext().getResources().getDisplayMetrics());
        //save data to share preferences
        SharedPreferences.Editor editor = getSharedPreferences("Settings", MODE_PRIVATE).edit();
        editor.putString("My_Lang", lang);
        editor.apply();
    }

    //load lang saved in shared p
    public void loadLocale() {
        // This method is used to load the saved language preference from SharedPreferences.
        SharedPreferences prefs = getSharedPreferences("Settings", Activity.MODE_PRIVATE);
        String language = prefs.getString("My_Lang", "");
        setLocale(language);
    }


    public void addImg(View view) {
        ImagePicker.with(MainActivity.this)
                .crop()                    //Crop image(Optional), Check Customization for more option
                .maxResultSize(1080, 1080)    //Final image resolution will be less than 1080 x 1080(Optional)
                .start();
    }

    // Data => User data
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        uri = data.getData(); // User selected image will be in data and it will return the
        // data in URI format

        // URI to bitmap
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        if (resultCode == Activity.RESULT_OK) {
            //Image Uri will not be null for RESULT_OK

            // Resizing our bitmap
            bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false);
            modelFunction(bitmap);

        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Pls Select Image", Toast.LENGTH_SHORT).show();
        }
    }

    public void modelFunction(Bitmap image) {
        try {

            FinalModel model = FinalModel.newInstance(MainActivity.this);

            // This line creates a fixed-size TensorBuffer object named inputFeature0. A TensorBuffer is a container for storing and manipulating multi-dimensional data, such as tensors used in deep learning models.
            //The new int[]{1, 224, 224, 3} parameter specifies the shape of the tensor:
            // 1 represents the batch size (1 image in this case), and 224x224x3 represents the dimensions of the image (224 pixels width, 224 pixels height, and 3 channels for RGB color).
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            //  ByteBuffer is like a container or a storage space that can hold a sequence of bytes.
            //  It's similar to a box where you can put different types of items, but in this case, the items are bytes of data.
            // This line creates a ByteBuffer object named byteBuffer, which is used to hold the image data in a byte format.
            //The allocateDirect() method allocates a direct byte buffer, which means it resides outside the normal garbage-collected heap and is more efficient for certain operations.
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            // The buffer size is calculated based on the imageSize (presumably the size of the image) and the fact that each pixel consists of 4 bytes (8 bits per channel for 3 color channels).
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            // This line creates an integer array named intValues with a size equal to imageSize squared.
            //The array will be used to hold the pixel values of the image.

            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            // This line extracts the pixel values from the image object and stores them in the intValues array.
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. convert them in float and Add those values individually to the byte buffer.
            // The purpose of the following step is to normalize the RGB values to the range of [0, 1] by dividing them by the maximum value of 255.
            // the code ensures that the input image data is properly formatted and compatible with the TFLite model, thus enabling accurate and reliable model inference.
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

           // The TFLite model is executed by calling the process method on the model object, passing the inputFeature0 tensor as input.
            //The inference result is stored in the outputs object of type FinalModel.Outputs.
            FinalModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // The output tensor, labeled as outputFeature0, is extracted from the outputs object as a TensorBuffer.
            //The TensorBuffer represents the output of the TFLite model after inference.
            float[] confidences = outputFeature0.getFloatArray();


            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxPos = i;
                    maxConfidence = confidences[i];
                }
            }

            String[] classes = {"Healthy", "Red rot", "Rust"};
            String diseaseName = classes[maxPos];

            // Navigating to Result Activity .
            Intent iPassData = new Intent(MainActivity.this, ResultActivity.class);
            iPassData.putExtra(USER_IMG_KEY, uri);
            iPassData.putExtra(DISEASE_NAME_KEY, diseaseName);
            startActivity(iPassData);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}