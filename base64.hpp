#ifndef BASE64_ENCODER
#define BASE64_ENCODER

/*
Adapted from Chris Venter's libb64 project, which has been placed in the public domain.
See http://sourceforge.net/projects/libb64 for the original code

C++ pure-header adaptation by Nicolas Brodu
Still in the public domain, do what you want with this

Usage (encoding):
-----------------

char* buffer_to_encode = new char[max_read_len];

base64 codec;
char* base64_encoded_text = new char[codec.get_max_encoded_size(max_read_len)];
int nread;
while (nread = some_read_function(buffer_to_encode, max_read_len)) {
    int nbytes = codec.encode(buffer_to_encode, nread, base64_encoded_text);
    some_write_function(base64_encoded_text, nbytes);
}
int nbytes = codec.encode_end(base64_encoded_text);
cout << base64_encoded_text << endl;

// call codec.reset_encoder() before encoding another stream with the same codec


Usage (decoding):
-----------------

char* base64_encoded_text = new char[max_read_len];

base64 codec;
char* decoded_buffer = new char[codec.get_max_decoded_size(max_read_len)];
int nread;
while (nread = some_read_function(base64_encoded_text, max_read_len)) {
    int nbytes = codec.decode(base64_encoded_text, nread, decoded_buffer);
    some_write_function(decoded_buffer, nbytes);
}

// call codec.reset_decoder() before decoding another stream with the same codec


Another example using a short string:
-------------------------------------

string params = ...; // contains the base64 encoded string

base64 codec;
vector<char> binary_params(codec.get_max_decoded_size(params.size()));
// no need for terminating 0, base64 has its own termination marker
codec.decode(params.c_str(), params.size(), &binary_params[0]);

*/

template<int _CHARS_PER_LINE>
struct base64_linebreak {

    typedef enum {
        step_A, step_B, step_C,
        step_a, step_b, step_c, step_d
    } Step;
    
    enum {
        CHARS_PER_LINE = _CHARS_PER_LINE
    };

    Step encodestep, decodestep;
    char encoderesult, decoderesult;
    int stepcount;

    inline base64_linebreak() : encodestep(step_A), decodestep(step_a), encoderesult(0), decoderesult(0), stepcount(0) {}

    inline void reset_encoder() {encodestep=step_A; encoderesult=0; stepcount=0;}
    inline void reset_decoder() {encodestep=step_a; decoderesult=0;}
    
    inline unsigned int get_max_encoded_size(int input_decoded_size) {
        int ret = input_decoded_size * 4 / 3; // 6 bits into 8 bits
        // add the line breaks, with support for some silly OS using 2 bytes per line break
        ret += (ret / CHARS_PER_LINE) * 2;
        // multiple of 4 bytes
        ret += (4 - (ret&3)) & 3;
        // space for final 0, just in case, and keep multiple of 4 too
        return ret + 4;
    }
    
    inline unsigned int get_max_decoded_size(int input_encoded_size) {
        // 8 bits -> 6 bits, rounded up
        return (input_encoded_size+1) * 3 / 4;
    }
    
    inline char encode_value(char value_in) const {
        static const char* encoding = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (value_in > 63) return '=';
        return encoding[(int)value_in];
    }

    inline int encode(const char* plaintext_in, int length_in, char* code_out) {
        const char* plainchar = plaintext_in;
        const char* const plaintextend = plaintext_in + length_in;
        char* codechar = code_out;
        char result;
        char fragment;
        
        result = encoderesult;
        
        switch (encodestep)
        {
            while (1)
            {
        case step_A:
                if (plainchar == plaintextend)
                {
                    encoderesult = result;
                    encodestep = step_A;
                    return codechar - code_out;
                }
                fragment = *plainchar++;
                result = (fragment & 0x0fc) >> 2;
                *codechar++ = encode_value(result);
                result = (fragment & 0x003) << 4;
        case step_B:
                if (plainchar == plaintextend)
                {
                    encoderesult = result;
                    encodestep = step_B;
                    return codechar - code_out;
                }
                fragment = *plainchar++;
                result |= (fragment & 0x0f0) >> 4;
                *codechar++ = encode_value(result);
                result = (fragment & 0x00f) << 2;
        case step_C:
                if (plainchar == plaintextend)
                {
                    encoderesult = result;
                    encodestep = step_C;
                    return codechar - code_out;
                }
                fragment = *plainchar++;
                result |= (fragment & 0x0c0) >> 6;
                *codechar++ = encode_value(result);
                result  = (fragment & 0x03f) >> 0;
                *codechar++ = encode_value(result);
                
                ++(stepcount);
                if (stepcount == CHARS_PER_LINE/4)
                {
                    *codechar++ = '\n';
                    stepcount = 0;
                }
            }
        }
        /* control should not reach here */
        return codechar - code_out;
    }
    
    int encode_end(char* code_out) {
        char* codechar = code_out;
        
        switch(encodestep)
        {
        case step_B:
            *codechar++ = encode_value(encoderesult);
            *codechar++ = '=';
            *codechar++ = '=';
            break;
        case step_C:
            *codechar++ = encode_value(encoderesult);
            *codechar++ = '=';
            break;
        case step_A:
            break;
        }
        *codechar++ = '\n';
        *codechar++ = 0;
        
        return codechar - code_out;
    }

    int decode_value(char value_in) {
        static const char decoding[] = {62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51};
        static const char decoding_size = sizeof(decoding);
        value_in -= 43;
        if (value_in < 0 || value_in > decoding_size) return -1;
        return decoding[(int)value_in];
    }

    int decode(const char* code_in, const int length_in, char* plaintext_out) {
        const char* codechar = code_in;
        char* plainchar = plaintext_out;
        char fragment;
        
        *plainchar = decoderesult;
        
        switch (decodestep)
        {
            while (1)
            {
        case step_a:
                do {
                    if (codechar == code_in+length_in)
                    {
                        decodestep = step_a;
                        decoderesult = *plainchar;
                        return plainchar - plaintext_out;
                    }
                    fragment = (char)decode_value(*codechar++);
                } while (fragment < 0);
                *plainchar    = (fragment & 0x03f) << 2;
        case step_b:
                do {
                    if (codechar == code_in+length_in)
                    {
                        decodestep = step_b;
                        decoderesult = *plainchar;
                        return plainchar - plaintext_out;
                    }
                    fragment = (char)decode_value(*codechar++);
                } while (fragment < 0);
                *plainchar++ |= (fragment & 0x030) >> 4;
                *plainchar    = (fragment & 0x00f) << 4;
        case step_c:
                do {
                    if (codechar == code_in+length_in)
                    {
                        decodestep = step_c;
                        decoderesult = *plainchar;
                        return plainchar - plaintext_out;
                    }
                    fragment = (char)decode_value(*codechar++);
                } while (fragment < 0);
                *plainchar++ |= (fragment & 0x03c) >> 2;
                *plainchar    = (fragment & 0x003) << 6;
        case step_d:
                do {
                    if (codechar == code_in+length_in)
                    {
                        decodestep = step_d;
                        decoderesult = *plainchar;
                        return plainchar - plaintext_out;
                    }
                    fragment = (char)decode_value(*codechar++);
                } while (fragment < 0);
                *plainchar++   |= (fragment & 0x03f);
            }
        }
        /* control should not reach here */
        return plainchar - plaintext_out;
    }

};

struct base64 : public base64_linebreak<72> {};

#endif
