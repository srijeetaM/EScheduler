using namespace std;


char *str_to_char_array(string s)
{
    char *split_liststring = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), split_liststring);
    split_liststring[s.size()] = '\0';
    return split_liststring;
}


//C++ method for splitting strings

class splitstring : public string {
    vector<string> flds;
public:
    splitstring(char *s) : string(s) { };
    vector<string>& split(char delim, int rep=0);
};

// split: receives a char delimiter; returns a vector of strings
// By default ignores repeated delimiters, unless argument rep == 1.
vector<string>& splitstring::split(char delim, int rep) {
    if (!flds.empty()) flds.clear();  // empty vector if necessary
    string work = data();
    string buf = "";
    int i = 0;
    while (i < work.length()) {
        if (work[i] != delim)
            buf += work[i];
        else if (rep == 1) {
            flds.push_back(buf);
            buf = "";
        } else if (buf.length() > 0) {
            flds.push_back(buf);
            buf = "";
        }
        i++;
    }
    if (!buf.empty())
        flds.push_back(buf);
    return flds;
}







// C Method for splitting
char **strsplit(const char *strn, const char *delim) {
    int i, j, count, len, l;
    char *str;
    char **list;
    list = (char **)NULL;
    //Make sure delim is a character
    if (delim[0]!='\0')
    {
        len = (strlen(strn) + 1);
        str = (char *)malloc(len * sizeof(char));
        strcpy(str,strn);
        for (i = 0; i < len; ++i)
        {
            for (j = 0; delim[j] != '\0'; ++j)
            {
                if(delim[j]==str[i])
                    str[i] = '\0'; 
            }
        }
        for (i = 0, count = 1; i < len; ++i)
        {
            if ((str[i] != '\0') && (str[i+1] == '\0'))
                count++;
        }
        // printf("%d\n", count);
        list = (char **)malloc(count * sizeof(char *));
        for (i = 0, j = 0; (i < count) && (j < len);)
        {
            if (str[j] == '\0') {
                j++;
                continue;
            }
            l = strlen(str + j) + 1;
            list[i] = (char *)malloc(l * sizeof(char));
            strcpy(list[i], str + j);
            j += l - 1;
            i++;
        }
        list[count - 1] = NULL;
        free(str);
    }
    return list;
}


void free_list(char **list) {

    int i;
    for (i = 0; list[i] != NULL; ++i)
        free(list[i]);
    free(list[i]);
    free(list);
    return;
}


int list_len(char **list) {
    int i;
    for (i = 0; list[i] != NULL; ++i);
    return i-1;
}

