package com.example.meatdetector_ta.view

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import com.example.meatdetector_ta.R
import com.theartofdev.edmodo.cropper.CropImage
import com.theartofdev.edmodo.cropper.CropImageView
import kotlinx.android.synthetic.main.fragment_crop.*
import java.io.File


class CropFragment : Fragment() {

    lateinit var uri:Uri
    lateinit var crop:Uri
    lateinit var bitmap: Bitmap

    lateinit var CropPath:String
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

    }

    override fun onCreateView(
            inflater: LayoutInflater, container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_crop, container, false)


    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayHomeAsUpEnabled(false)
        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayShowHomeEnabled(false)

        var path = CropFragmentArgs.fromBundle(requireArguments()).Path
        var tipe = CropFragmentArgs.fromBundle(requireArguments()).tipe

        if(tipe =="kamera") {
            val f = File(path)
            uri = Uri.fromFile(f)
        }
        else if(tipe == "galeri")
        {
            try{
                uri = Uri.parse(path)
            }
            catch (e: NullPointerException){
                Toast.makeText(activity, e.toString(), Toast.LENGTH_LONG).show()
            }
        }

        launchImagecrop(uri)

        buttonDeteksi.setOnClickListener{
            try {
                val action = CropFragmentDirections.actionKlasifikasiFragment(CropPath)
                Navigation.findNavController(it).navigate(action)

            }
            catch (e: InterruptedException){
                Toast.makeText(activity, e.toString(), Toast.LENGTH_LONG).show()
            }
        }

        buttonBatal.setOnClickListener{
            val action = CropFragmentDirections.actionMainFragment()
            Navigation.findNavController(it).navigate(action)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        try {
            if (requestCode === CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {

                var result = CropImage.getActivityResult(data)
                imageCropDaging.setImageURI(result.uri)

                CropPath = result.uri.toString()
            }
        }
        catch (e: NullPointerException){
            Toast.makeText(activity, "Gagal Melakukan Crop. Mohon Melakukan Upload image Ulang", Toast.LENGTH_LONG).show()
        }
    }

    private fun launchImagecrop(uri: Uri)
    {
        var view = view
        CropImage.activity(uri)
                .setGuidelines(CropImageView.Guidelines.ON)
                .setAspectRatio(200, 200)
                .setCropShape(CropImageView.CropShape.RECTANGLE)
                .start(view!!.context, this)
    }


}