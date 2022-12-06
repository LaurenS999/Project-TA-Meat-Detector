package com.example.meatdetector_ta.view

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import com.example.meatdetector_ta.R
import com.example.meatdetector_ta.ml.HyperparameterDensenet169NadamA30L90
import com.example.meatdetector_ta.ml.Model5MobilenetNadam
import kotlinx.android.synthetic.main.fragment_klasifikasi.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.text.DecimalFormat
import java.util.*


class KlasifikasiFragment : Fragment() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
            inflater: LayoutInflater, container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_klasifikasi, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayHomeAsUpEnabled(false)
        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayShowHomeEnabled(false)

        var path = KlasifikasiFragmentArgs.fromBundle(requireArguments()).Path
        var uri = Uri.parse(path)
        imageCitraFinal.setImageURI(uri)

        Klasifikasi("DenseNet")

        buttonKembali.setOnClickListener{
            val action = KlasifikasiFragmentDirections.actionKlasifikasiFragmentToMainFragment()
            Navigation.findNavController(it).navigate(action)
        }

        radioButtonDensenet.setOnClickListener { Klasifikasi("DenseNet") }
        radioButtonMobileNet.setOnClickListener { Klasifikasi("MobileNet") }
    }

    fun Klasifikasi(jenisModel: String) {

        var ModelMobileNet = Model5MobilenetNadam.newInstance(requireContext())
        var ModelDenseNet = HyperparameterDensenet169NadamA30L90.newInstance(requireContext())

        var image= imageCitraFinal
        var bitmap = image.getDrawable().toBitmap()
        bitmap = Bitmap.createScaledBitmap(bitmap, 200, 200, true)

        var imageprocessor = ImageProcessor.Builder()
            .add(ResizeOp(200, 200, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(0.0f, 255.0f))
             .build()

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 200, 200, 3), DataType.FLOAT32)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        var TensorNormalize = imageprocessor.process(tensorImage)
        val byteBuffer = TensorNormalize.buffer

        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.

        if(jenisModel =="MobileNet")
        {
            val outputsMobileNet = ModelMobileNet.process(inputFeature0)
            val outputFeature0MobileNet = outputsMobileNet.outputFeature0AsTensorBuffer

            var max = getMax(outputFeature0MobileNet.floatArray)
            var kategori = arrayOf<String>("Daging Ayam Segar", "Daging Ayam Tidak Segar", "Daging Babi Segar",
                    "Daging Babi Tidak Segar", "Daging Bebek Segar", "Daging Bebek Tidak Segar",
                    "Daging Kambing Segar", "Daging Kambing Tidak Segar", "Daging Sapi Segar",
                    "Daging Sapi Tidak Segar")

            var df = DecimalFormat("#.##")
            var presentasi = df.format(outputFeature0MobileNet.floatArray[max] * 100.0)

            if(outputFeature0MobileNet.floatArray[max] * 100.0 >= 30) {
                txtOuputJenis.setText(kategori[max])
                txtOuputPersentasi.setText(presentasi.toString() + "%")
            }
            else
            {
                txtOuputJenis.setText("Object tidak terdeteksi. Mohon input gambar ulang")
                txtOuputPersentasi.setText("-")
            }
        }
        else if(jenisModel == "DenseNet")
        {
            val outputsDenseNet = ModelDenseNet.process(inputFeature0)
            val outputFeature0DenseNet = outputsDenseNet.outputFeature0AsTensorBuffer

            var max = getMax(outputFeature0DenseNet.floatArray)
            var kategori = arrayOf<String>("Daging Ayam Segar", "Daging Ayam Tidak Segar", "Daging Babi Segar",
                    "Daging Babi Tidak Segar", "Daging Bebek Segar", "Daging Bebek Tidak Segar",
                    "Daging Kambing Segar", "Daging Kambing Tidak Segar", "Daging Sapi Segar",
                    "Daging Sapi Tidak Segar")

            var df = DecimalFormat("#.##")
            var presentasi = df.format(outputFeature0DenseNet.floatArray[max] * 100.0)

            if(outputFeature0DenseNet.floatArray[max] * 100.0 >= 30) {
                txtOuputJenis.setText(kategori[max])
                txtOuputPersentasi.setText(presentasi.toString() + "%")
            }
            else
            {
                txtOuputJenis.setText("Object tidak terdeteksi. Mohon input gambar ulang")
                txtOuputPersentasi.setText("-")
            }
        }

        // Releases model resources if no longer used.
        ModelMobileNet.close()
        ModelDenseNet.close()
    }

    fun getMax(arr: FloatArray) :Int {
        var ind =0
        var min = 0.0f
        for(i in 0..9)
        {
            if(arr[i] >min)
            {
                ind=i
                min = arr[i]
            }
        }
        return ind
    }
}