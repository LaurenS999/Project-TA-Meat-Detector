package com.example.meatdetector_ta.view

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.meatdetector_ta.R
import com.example.meatdetector_ta.viewmodel.InformasiModel
import kotlinx.android.synthetic.main.fragment_informasi.*

class InformasiFragment : Fragment() {
    private lateinit var viewModel:InformasiModel
    private val informasiAdapter = informasiAdapter(arrayListOf())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_informasi, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayHomeAsUpEnabled(false)
        (activity as AppCompatActivity?)!!.supportActionBar!!.setDisplayShowHomeEnabled(false)

        viewModel = ViewModelProvider(this).get(InformasiModel::class.java)
        viewModel.refresh()

        RecViewInformasi.layoutManager = LinearLayoutManager(context)
        RecViewInformasi.adapter = informasiAdapter

        observeViewModel()
    }

    fun observeViewModel()
    {
        viewModel.informasiLD.observe(viewLifecycleOwner, Observer{
            informasiAdapter.refreshData(it)
        })
    }

}