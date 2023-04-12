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
import android.net.Uri;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.github.dhaval2404.imagepicker.ImagePicker;

import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    static final String USER_IMG_KEY = "UserImgKey";
    static final int REQ_CODE = 100 ;
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
        final String[] listItems = {"हिंदी" , "मराठी", "English"}; //array of languages
        AlertDialog.Builder mBuilder = new AlertDialog.Builder(MainActivity.this);
        mBuilder.setTitle("Choose Language...");
        mBuilder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                if (i == 0)
                {
                    setLocale("hi");
                    recreate();
                }
                if (i == 1)
                {
                    setLocale("mr");
                    recreate();
                }
                if (i == 2)
                {
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

        if (resultCode == Activity.RESULT_OK) {
            //Image Uri will not be null for RESULT_OK
            // Passing Data
            Intent iPassData = new Intent(MainActivity.this , ResultActivity.class);
            iPassData.putExtra(USER_IMG_KEY , uri);
            startActivity(iPassData);
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Pls Select Image", Toast.LENGTH_SHORT).show();
        }
    }
}