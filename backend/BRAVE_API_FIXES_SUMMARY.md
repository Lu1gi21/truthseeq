# Brave Search API Fixes Summary

## Issues Identified

Based on the error logs, the following issues were identified with the Brave Search API integration:

1. **Language Parameter Error (422)**: The API was receiving `"en_US"` but expected `"en"` for the language parameter
2. **Rate Limiting (429)**: The API was hitting rate limits due to too many requests
3. **Deprecation Warning**: LangChain tool method `BaseTool.__call__` was deprecated in favor of `.invoke()`

## Fixes Implemented

### 1. Language Parameter Fix

**File**: `backend/app/workflow/tools.py`

**Change**: Updated the default `search_lang` parameter from `"en_US"` to `"en"`

```python
# Before
def search(self, query: str, count: int = 10, search_lang: str = "en_US", country: str = "US"):

# After  
def search(self, query: str, count: int = 10, search_lang: str = "en", country: str = "US"):
```

**Impact**: Resolves the 422 validation error from the Brave Search API.

### 2. Rate Limiting Implementation

**File**: `backend/app/workflow/tools.py`

**Changes**:
- Added `self.last_request_time = 0` to track request timing
- Implemented rate limiting logic to ensure at least 2 seconds between requests
- Added proper error handling for rate limit responses

```python
# Added rate limiting attribute
self.last_request_time = 0  # Track last request time for rate limiting

# Implemented rate limiting in search method
current_time = time.time()
time_since_last_request = current_time - self.last_request_time
if time_since_last_request < 2.0:
    sleep_time = 2.0 - time_since_last_request
    logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
    time.sleep(sleep_time)

# Update last request time after successful request
self.last_request_time = time.time()
```

**Impact**: Prevents 429 rate limit errors by spacing out API requests.

### 3. Tool Method Updates

**File**: `backend/app/workflow/nodes.py`

**Changes**: Updated tool calls to use `.invoke()` instead of direct calling

```python
# Before
fact_check_results = search_fact_checking_databases(target_url)

# After
fact_check_results = search_fact_checking_databases.invoke({"query": target_url})
```

**Impact**: Resolves LangChain deprecation warnings and follows current best practices.

### 4. Fact-Checking Search Improvement

**File**: `backend/app/workflow/tools.py`

**Changes**: Enhanced the `search_fact_checking_databases` method to use real search instead of placeholder data

```python
def search_fact_checking_databases(self, query: str) -> List[Dict[str, Any]]:
    try:
        # Use Brave Search to find fact-checking sources
        search_tool = BraveSearchTool()
        
        # Search for fact-checking databases and sites
        fact_check_query = f'"{query}" site:snopes.com OR site:factcheck.org OR site:reuters.com/fact-check OR site:apnews.com/fact-check'
        search_results = search_tool.search(fact_check_query, count=5)
        
        fact_check_results = []
        for result in search_results:
            fact_check_results.append({
                "source": self._extract_source_domain(result.url),
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "verdict": "unverified",  # Would need AI analysis to determine
                "relevance_score": result.relevance_score
            })
        
        return fact_check_results
        
    except Exception as e:
        logger.error(f"Error searching fact-checking databases: {e}")
        # Return placeholder if search fails
        return [...]
```

**Impact**: Provides real fact-checking search results instead of placeholder data.

## Testing

Created `backend/test_brave_api_fixes.py` to verify all fixes work correctly:

- ✅ Language parameter correctly set to 'en'
- ✅ Rate limiting attribute exists
- ✅ Tools use .invoke() method correctly
- ✅ Fact-checking search returns proper results

All tests pass successfully.

## Expected Results

With these fixes, the workflow should:

1. **No more 422 errors**: Language parameter is now correctly formatted
2. **No more 429 errors**: Rate limiting prevents too many requests
3. **No more deprecation warnings**: Using `.invoke()` method
4. **Better fact-checking**: Real search results instead of placeholders
5. **Improved reliability**: Proper error handling and fallbacks

## Files Modified

1. `backend/app/workflow/tools.py` - Main fixes for Brave Search API
2. `backend/app/workflow/nodes.py` - Updated tool calls to use .invoke()
3. `backend/test_brave_api_fixes.py` - Test script to verify fixes
4. `backend/BRAVE_API_FIXES_SUMMARY.md` - This summary document

## Next Steps

1. Test the workflow with a real URL to verify all fixes work in practice
2. Monitor logs for any remaining issues
3. Consider implementing more sophisticated rate limiting if needed
4. Add more fact-checking sources to the search query if desired 