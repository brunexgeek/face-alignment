

/* boost random/mersenne_twister.hpp header file
 *
 * Copyright Jens Maurer 2000-2001
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * See http://www.boost.org for most recent version including documentation.
 *
 * $Id: mersenne_twister.hpp,v 1.20 2005/07/21 22:04:31 jmaurer Exp $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
*/

#ifndef DLIB_BOOST_RANDOM_MERSENNE_TWISTER_HPP
#define DLIB_BOOST_RANDOM_MERSENNE_TWISTER_HPP

#include <iostream>
#include <algorithm>     // std::copy
#include <stdexcept>
//#include "../uintn.h"
//#include "../serialize.h"

namespace ert
{
    namespace random_helpers
    {

    // ------------------------------------------------------------------------------------

        // http://www.math.keio.ac.jp/matumoto/emt.html
        template<
            class UIntType,
            int w,
            int n,
            int m,
            int r,
            UIntType a,
            int u,
            int s,
            UIntType b,
            int t,
            UIntType c,
            int l,
            UIntType val
            >
        class mersenne_twister
        {
        public:
            typedef UIntType result_type;
            const static int word_size = w;
            const static int state_size = n;
            const static int shift_size = m;
            const static int mask_bits = r;
            const static UIntType parameter_a = a;
            const static int output_u = u;
            const static int output_s = s;
            const static UIntType output_b = b;
            const static int output_t = t;
            const static UIntType output_c = c;
            const static int output_l = l;

            const static bool has_fixed_range = false;

            mersenne_twister() { seed(); }

            explicit mersenne_twister(UIntType value) { seed(value); }

            void seed () { seed(UIntType(5489)); }

            // compiler-generated copy ctor and assignment operator are fine

            void seed(UIntType value)
            {
                // New seeding algorithm from
                // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
                // In the previous versions, MSBs of the seed affected only MSBs of the
                // state x[].
                const UIntType mask = ~0u;
                x[0] = value & mask;
                for (i = 1; i < n; i++) {
                    // See Knuth "The Art of Computer Programming" Vol. 2, 3rd ed., page 106
                    x[i] = (1812433253UL * (x[i-1] ^ (x[i-1] >> (w-2))) + i) & mask;
                }
            }

            result_type min() const { return 0; }
            result_type max() const
            {
                // avoid "left shift count >= with of type" warning
                result_type res = 0;
                for(int i = 0; i < w; ++i)
                    res |= (1u << i);
                return res;
            }

            result_type operator()();

            /*friend void serialize(
                const mersenne_twister& item,
                std::ostream& out
            )
            {
                dlib::serialize(item.x, out);
                dlib::serialize(item.i, out);
            }

            friend void deserialize(
                mersenne_twister& item,
                std::istream& in
            )
            {
                dlib::deserialize(item.x, in);
                dlib::deserialize(item.i, in);
            }*/

        private:

            void twist(int block);

            // state representation: next output is o(x(i))
            //   x[0]  ... x[k] x[k+1] ... x[n-1]     x[n]     ... x[2*n-1]   represents
            //  x(i-k) ... x(i) x(i+1) ... x(i-k+n-1) x(i-k-n) ... x[i(i-k-1)]
            // The goal is to always have x(i-n) ... x(i-1) available for
            // operator== and save/restore.

            UIntType x[2*n];
            int i;
        };

    // ------------------------------------------------------------------------------------

        template<
            class UIntType, int w, int n, int m, int r, UIntType a, int u,
            int s, UIntType b, int t, UIntType c, int l, UIntType val
            >
        void mersenne_twister<UIntType,w,n,m,r,a,u,s,b,t,c,l,val>::twist(
            int block
        )
        {
            const UIntType upper_mask = (~0u) << r;
            const UIntType lower_mask = ~upper_mask;

            if(block == 0) {
                for(int j = n; j < 2*n; j++) {
                    UIntType y = (x[j-n] & upper_mask) | (x[j-(n-1)] & lower_mask);
                    x[j] = x[j-(n-m)] ^ (y >> 1) ^ (y&1 ? a : 0);
                }
            } else if (block == 1) {
                // split loop to avoid costly modulo operations
                {  // extra scope for MSVC brokenness w.r.t. for scope
                    for(int j = 0; j < n-m; j++) {
                        UIntType y = (x[j+n] & upper_mask) | (x[j+n+1] & lower_mask);
                        x[j] = x[j+n+m] ^ (y >> 1) ^ (y&1 ? a : 0);
                    }
                }

                for(int j = n-m; j < n-1; j++) {
                    UIntType y = (x[j+n] & upper_mask) | (x[j+n+1] & lower_mask);
                    x[j] = x[j-(n-m)] ^ (y >> 1) ^ (y&1 ? a : 0);
                }
                // last iteration
                UIntType y = (x[2*n-1] & upper_mask) | (x[0] & lower_mask);
                x[n-1] = x[m-1] ^ (y >> 1) ^ (y&1 ? a : 0);
                i = 0;
            }
        }

    // ------------------------------------------------------------------------------------

        template<
            class UIntType, int w, int n, int m, int r, UIntType a, int u,
            int s, UIntType b, int t, UIntType c, int l, UIntType val
            >
        inline typename mersenne_twister<UIntType,w,n,m,r,a,u,s,b,t,c,l,val>::result_type
        mersenne_twister<UIntType,w,n,m,r,a,u,s,b,t,c,l,val>::operator()(
        )
        {
            if(i == n)
                twist(0);
            else if(i >= 2*n)
                twist(1);
            // Step 4
            UIntType z = x[i];
            ++i;
            z ^= (z >> u);
            z ^= ((z << s) & b);
            z ^= ((z << t) & c);
            z ^= (z >> l);
            return z;
        }

    // ------------------------------------------------------------------------------------

    } // namespace random


    typedef random_helpers::mersenne_twister<uint32_t,32,351,175,19,0xccab8ee7,11,
    7,0x31b6ab00,15,0xffe50000,17, 0xa37d3c92> mt11213b;

    // validation by experiment from mt19937.c
    typedef random_helpers::mersenne_twister<uint32_t,32,624,397,31,0x9908b0df,11,
    7,0x9d2c5680,15,0xefc60000,18, 3346425566U> mt19937;


    class Random
    {

        /*!
            INITIAL VALUE
                - seed == ""

            CONVENTION
                - the random numbers come from the boost mersenne_twister code
                - get_seed() == seed
        !*/

        public:


            Random(
            )
            {
                init();
            }

            Random (
                const std::string& seed_value
            )
            {
                init();
                set_seed(seed_value);
            }

            virtual ~Random(
            )
            {}

            void clear(
            )
            {
                mt.seed();
                seed.clear();

                has_gaussian = false;
                next_gaussian = 0;

                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();
            }

            const std::string& get_seed (
            )
            {
                return seed;
            }

            void set_seed (
                const std::string& value
            )
            {
                seed = value;

                // make sure we do the seeding so that using a seed of "" gives the same
                // state as calling this->clear()
                if (value.size() != 0)
                {
                    uint32_t s = 0;
                    for (std::string::size_type i = 0; i < seed.size(); ++i)
                    {
                        s = (s*37) + static_cast<uint32_t>(seed[i]);
                    }
                    mt.seed(s);
                }
                else
                {
                    mt.seed();
                }

                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();


                has_gaussian = false;
                next_gaussian = 0;
            }

            unsigned char get_random_8bit_number (
            )
            {
                return static_cast<unsigned char>(mt());
            }

            uint16_t get_random_16bit_number (
            )
            {
                return static_cast<uint16_t>(mt());
            }

            inline uint32_t get_random_32bit_number (
            )
            {
                return mt();
            }

            inline uint64 get_random_64bit_number (
            )
            {
                const uint64 a = get_random_32bit_number();
                const uint64 b = get_random_32bit_number();
                return (a<<32)|b;
            }

            double get_random_double (
            )
            {
                uint32_t temp;

                temp = Random::get_random_32bit_number();
                temp &= 0xFFFFFF;

                double val = static_cast<double>(temp);

                val *= 0x1000000;

                temp = Random::get_random_32bit_number();
                temp &= 0xFFFFFF;

                val += temp;

                val /= max_val;

                if (val < 1.0)
                {
                    return val;
                }
                else
                {
                    // return a value slightly less than 1.0
                    return 1.0 - std::numeric_limits<double>::epsilon();
                }
            }

            float get_random_float (
            )
            {
                uint32_t temp;

                temp = Random::get_random_32bit_number();
                temp &= 0xFFFFFF;

                const float scale = 1.0/0x1000000;

                const float val = static_cast<float>(temp)*scale;
                if (val < 1.0f)
                {
                    return val;
                }
                else
                {
                    // return a value slightly less than 1.0
                    return 1.0f - std::numeric_limits<float>::epsilon();
                }
            }

            double get_random_gaussian (
            )
            {
                if (has_gaussian)
                {
                    has_gaussian = false;
                    return next_gaussian;
                }

                double x1, x2, w;

                const double rndmax = std::numeric_limits<uint32_t>::max();

                // Generate a pair of Gaussian random numbers using the Box-Muller transformation.
                do
                {
                    const double rnd1 = get_random_32bit_number()/rndmax;
                    const double rnd2 = get_random_32bit_number()/rndmax;

                    x1 = 2.0 * rnd1 - 1.0;
                    x2 = 2.0 * rnd2 - 1.0;
                    w = x1 * x1 + x2 * x2;
                } while ( w >= 1.0 );

                w = std::sqrt( (-2.0 * std::log( w ) ) / w );
                next_gaussian = x2 * w;
                has_gaussian = true;
                return x1 * w;
            }

            /*void swap (
                rand& item
            )
            {
                exchange(mt,item.mt);
                exchange(seed, item.seed);
                exchange(has_gaussian, item.has_gaussian);
                exchange(next_gaussian, item.next_gaussian);
            }*/

            friend void serialize(
                const Random& item,
                std::ostream& out
            );

            friend void deserialize(
                Random& item,
                std::istream& in
            );

        private:

            void init()
            {
                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();

                max_val =  0xFFFFFF;
                max_val *= 0x1000000;
                max_val += 0xFFFFFF;
                max_val += 0.01;


                has_gaussian = false;
                next_gaussian = 0;
            }

            mt19937 mt;

            std::string seed;


            double max_val;
            bool has_gaussian;
            double next_gaussian;
    };

}


#endif // DLIB_BOOST_RANDOM_MERSENNE_TWISTER_HPP

