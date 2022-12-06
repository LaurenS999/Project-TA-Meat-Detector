package com.example.meatdetector_ta.view

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import com.example.meatdetector_ta.R
import kotlinx.android.synthetic.main.fragment_main.*
import java.io.File

class MainFragment : Fragment() {

    private val FILE_NAME ="photo"
    private val REQUEST_CODE = 42
    private lateinit var photoFile: File

    lateinit var path:String
    lateinit var tipe:String
    private lateinit var output:File

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
            inflater: LayoutInflater, container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_main, container, false)

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayHomeAsUpEnabled(false)
        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayShowHomeEnabled(false)

        buttonLanjut.setOnClickListener{
            val action = MainFragmentDirections.actionCropFragment(path,tipe)
            Navigation.findNavController(it).navigate(action)
        }

        buttoncamera.setOnClickListener(View.OnClickListener {
            try {
                val takepicktureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                photoFile = getPhotoFile(FILE_NAME)

                val fileprovider = FileProvider.getUriForFile(view.context, "com.example.meatdetector_ta.fileprovider", photoFile)
                takepicktureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileprovider)
                startActivityForResult(takepicktureIntent, REQUEST_CODE)

            } catch (e: NullPointerException) {
                Toast.makeText(activity, e.toString(), Toast.LENGTH_LONG).show()
            }
        })

        buttonGaleri.setOnClickListener {
            try {
                var galeriintent = Intent(Intent.ACTION_PICK)
                galeriintent.type = "image/*"
                startActivityForResult(galeriintent, 200)
            }
            catch (e: NullPointerException){
                Toast.makeText(activity, e.toString(), Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun getPhotoFile(fileName:String): File{
        val storageDirectory = activity?.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(fileName, ".jpg", storageDirectory)
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK  && data != null){
            if(requestCode == 200 ) {
                imageCitraDaging.setImageURI(data?.data)

                path = data?.data.toString()
                tipe = "galeri"
            }
        }

        if(requestCode == REQUEST_CODE && resultCode == Activity.RESULT_OK){
            val takenImage = BitmapFactory.decodeFile(photoFile.absolutePath)
            imageCitraDaging.setImageBitmap(takenImage)
            path = photoFile.absolutePath
            tipe= "kamera"

        }
    }
}