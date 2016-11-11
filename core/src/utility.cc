// The MIT License (MIT)
// 
// Copyright (c) 2016 Northeastern University
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "utility.h"

namespace dnnmark {

void TrimStr(std::string *s) {
  TrimStrLeft(s);
  TrimStrRight(s);
}

void TrimStrLeft(std::string *s) {
  s->erase(s->begin(), std::find_if(s->begin(), s->end(),
           std::not1(std::ptr_fun<int, int>(std::isspace))));
}

void TrimStrRight(std::string *s) {
  s->erase(std::find_if(s->rbegin(), s->rend(),
           std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s->end());
}

void SplitStr(const std::string &s, std::string *var, std::string *val,
              std::string delimiter = "=") {
  // Obtain the position of equal sign
  std::size_t pos = s.find_first_of(delimiter);

  // TODO: Add an error detetion here

  // Obtain the substring of variable and value
  *var = s.substr(0, pos-1);
  *val = s.substr(pos+1, std::string::npos);
}

bool isCommentStr(const std::string &s, char comment_marker = "#") {
  TrimStr(&s);
  return s[0] == comment_marker;
}

} // namespace dnnmark

