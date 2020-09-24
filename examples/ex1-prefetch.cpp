//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/util/forall.hpp"

#define REPS 1
#define SIZE 50

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{
  /*
   * Allocate an array.
   */
  chai::ManagedArray<double> array(SIZE);

  forall(sequential(), 0, SIZE, [=](int i) { array[i] = 1.0f; });

  /*
   * Fill data on the device
   */
  for(int j=0; j<REPS; j++) {
#if defined(PREFETCH)
      std::cout << "Prefetching..." << std::endl;
      // Prefetch
      array.move(chai::GPU);
#endif

      std::cout << "Start kernels..." << std::endl;
      forall(gpu(), 0, SIZE, [=] __device__(int i) { array[i] = i * 2.0f; });

      std::cout << "End kernels..." << std::endl;

      /*
       * Print the array on the host, data is automatically copied back.
       */
      std::cout << "array = [";
      forall(sequential(), 0, 10, [=](int i) { std::cout << " " << array[i]; });
      std::cout << " ]" << std::endl;
  }
}
