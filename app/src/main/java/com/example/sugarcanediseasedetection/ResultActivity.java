package com.example.sugarcanediseasedetection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

public class ResultActivity extends AppCompatActivity {
    ImageView diseaseImg;
    TextView diseaseName;
    Button infoBtn;
    String DiseaseName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        diseaseImg = findViewById(R.id.diseaseImg);
        diseaseName = findViewById(R.id.diseaseName);
        infoBtn = findViewById(R.id.infoBtn);

        // Receiving User Img
        Intent iGetData = getIntent();
        Uri userImg = (Uri) iGetData.getParcelableExtra(MainActivity.USER_IMG_KEY);
        diseaseImg.setImageURI(userImg);

        // Disease Name
        diseaseName.setText(getIntent().getStringExtra(MainActivity.DISEASE_NAME_KEY));

        // Edge Case : If inappropriate disease is

        // InfoBtn Logic
        // We'll extract the text from diseaseName and apply onClick on InfoBtn and
        // using switch case we'll show particular detected disease info

        DiseaseName = (String) diseaseName.getText();
        infoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (DiseaseName.equals("Red rot")) {
                    Intent goToInfo = new Intent(ResultActivity.this, RedStrip.class);
                    startActivity(goToInfo);
                    finish();
                } else if (DiseaseName.equals("Leaf Scald")) {
                    Intent goToInfo = new Intent(ResultActivity.this, LeafScald.class);
                    startActivity(goToInfo);
                    finish();
                } else if (DiseaseName.equals("Rust")) {
                    Intent goToInfo = new Intent(ResultActivity.this, Rust.class);
                    startActivity(goToInfo);
                    finish();
                } else if (DiseaseName.equals("Healthy")) {
                    infoBtn.setEnabled(false);
                    Toast.makeText(ResultActivity.this, "Your Leaf is  Healthy",
                            Toast.LENGTH_SHORT).show();
                    finish();
                } else {
                    infoBtn.setEnabled(false);
                    Toast.makeText(ResultActivity.this, "Inappropriate Disease",
                            Toast.LENGTH_SHORT).show();
                    finish();
                }
            }
        });
    }
}