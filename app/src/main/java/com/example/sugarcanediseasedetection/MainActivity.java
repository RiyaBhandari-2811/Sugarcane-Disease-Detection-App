package com.example.sugarcanediseasedetection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.example.sugarcanediseasedetection.ml.Model;
import com.github.dhaval2404.imagepicker.ImagePicker;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    static final String USER_IMG_KEY = "UserImgKey";
    static final int REQ_CODE = 100;
    Bitmap bitmap;
    TextView result;

    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        loadLocale();
        setContentView(R.layout.activity_main);

        Button changelng = findViewById(R.id.changeMyLang);
        result = (TextView) findViewById(R.id.resultt);
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
        mBuilder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
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
        mDialog.show();
    }

    private void setLocale(String lang) {
        Locale locale = new Locale(lang);
        Locale.setDefault(locale);
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
        Uri uri = data.getData(); // User selected image will be in data and it will return the
        // data in URI format

        // URI to bitmap
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        if (resultCode == Activity.RESULT_OK) {
            //Image Uri will not be null for RESULT_OK
            // Passing Data
//            Intent iPassData = new Intent(MainActivity.this , ResultActivity.class);
//            iPassData.putExtra(USER_IMG_KEY , uri);
//            startActivity(iPassData);
            // Resizing our bitmap
            bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false);
            modelFunction(bitmap);
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Pls Select Image", Toast.LENGTH_SHORT).show();
        }
    }

    public void modelFunction(Bitmap bitmap) {
        try {
            Model model = Model.newInstance(MainActivity.this);


            // Creates inputs for reference.
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[imageSize * imageSize];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            int pixels = 0;
// Iterate over each pixel and extract RGB values. Add those values individually to the byte buffer.
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixels++]; // RGB Values
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);


            // Our result are stored in outputFeature0
            Model.Outputs outputs = model.process(inputFeature0);
            // Runs model inference and gets result.
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

           float[] confidences = outputFeature0.getFloatArray();
           // Find the index of the class with the biggest confidence .
            int maxPos = 0;
            float maxConfidence = 0 ;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i ;
                }
            }

            // Labels that our model is trained for
            String[] labels = {"Red Strip" , "Healthy" , "Rust"};
            result.setText(labels[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

}