# Phase 3: Testing Guide - Predictive Analytics

## 🎯 What to Test

You've successfully implemented Phase 3! Now let's test everything end-to-end.

---

## ✅ Pre-Testing Checklist

### Backend
- [x] Backend server running on port 8000
- [x] OpenWeatherMap API key added to `.env`
- [x] Database is accessible
- [x] All new services created

### Frontend
- [ ] Frontend running on port 5173 (or 3000)
- [ ] New components compiled successfully
- [ ] Routes added to App.tsx

---

## 🚀 Start the Application

### Terminal 1: Backend
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2: Frontend
```bash
cd frontend/web
npm run dev
```

Access: **http://localhost:5173** (or http://localhost:3000)

---

## 📋 Testing Scenarios

### Scenario 1: View Weather on Farm Detail Page

**Steps:**
1. Login to your account
2. Navigate to **Farms** page
3. Click on any farm (that has GPS coordinates)
4. **Expected Results:**
   - You should see a compact weather widget below the farm stats
   - Current temperature, humidity, wind speed displayed
   - Weather icon and description
   - If no API key: "Mock Data" badge appears

**Success Criteria:**
- ✅ Weather widget loads without errors
- ✅ Data displays correctly
- ✅ Weather is relevant to farm location

---

### Scenario 2: Access Planting Recommendations

**Steps:**
1. From Farms page, click the **"Planting"** button on any farm card
   OR
2. From Farm Detail page, click **"Planting Recommendations"** button

**Expected Results:**
- Redirects to `/farms/{farmId}/planting`
- Shows comparison view with 6 crops:
  - Maize
  - Rice
  - Cassava
  - Tomato
  - Soybean
  - Groundnut
- Each crop shows:
  - Suitability score (0-100)
  - Recommendation badge (Highly Recommended / Recommended / Not Recommended)
  - Planting date
  - Harvest date
  - Summary

**Success Criteria:**
- ✅ Page loads successfully
- ✅ All 6 crops displayed
- ✅ Scores are calculated
- ✅ Recommendations are appropriate

---

### Scenario 3: View Detailed Crop Recommendation

**Steps:**
1. On Planting Recommendations page
2. Click on any crop card (e.g., "Maize")

**Expected Results:**
- Switches to detailed view
- Shows comprehensive analysis:
  1. **Overall Score Card**
     - Large suitability score
     - Recommendation status
     - Summary with contributing factors

  2. **Planting Window**
     - Earliest, Recommended, Latest dates
     - Reason for timing
     - Expected harvest date

  3. **Weather Analysis**
     - Average temperature
     - Total rainfall (7 days)
     - Rainy days count
     - Favorable conditions list
     - Concerns list

  4. **Seasonal Analysis**
     - Season status (optimal/good/off-season)
     - Planting months display

  5. **Soil Analysis**
     - Soil suitability level
     - Strengths and concerns

  6. **Preparation Checklist**
     - General tasks
     - Crop-specific tasks

**Success Criteria:**
- ✅ All sections display
- ✅ Data is accurate and relevant
- ✅ Recommendations make sense
- ✅ Can return to comparison view

---

### Scenario 4: Test Different Crops

**Steps:**
1. View recommendations for different crops
2. Compare suitability scores

**Expected Behavior:**
- Scores differ based on:
  - Current weather
  - Season (month of year)
  - Soil conditions
  - ML model predictions

**Example Expected Results (November in Ghana):**
- **Maize**: ~85/100 (Primary season)
- **Rice**: ~75/100 (Good season)
- **Cassava**: ~80/100 (Can plant anytime)
- **Tomato**: ~70/100 (Approaching season)

---

### Scenario 5: Test Weather API Integration

#### With OpenWeatherMap API Key:
**Expected:**
- Real-time weather data
- Current conditions for farm location
- Accurate temperature/humidity
- No "Mock Data" warning

#### Without API Key:
**Expected:**
- Mock weather data displays
- "Mock Data" badge visible
- Temperature: ~24.5°C
- Humidity: ~65%
- System still works (graceful degradation)

---

### Scenario 6: Test Error Handling

#### Test 1: Farm Without GPS
**Steps:**
1. Create a farm without latitude/longitude
2. Try to view planting recommendations

**Expected:**
- Error message: "Farm must have GPS coordinates"
- Graceful error handling

#### Test 2: Backend Down
**Steps:**
1. Stop backend server
2. Try to load weather or recommendations

**Expected:**
- Loading state
- Error message displayed
- "Retry" button appears

---

## 🔍 API Testing (Swagger UI)

Visit: **http://localhost:8000/docs**

### Test Endpoints:

#### 1. Weather Forecast
```
GET /api/recommendations/weather/1
```
**Expected Response:**
```json
{
  "success": true,
  "farm_id": 1,
  "current_weather": {
    "temperature": 24.5,
    "humidity": 65,
    ...
  },
  "forecast": {
    "forecast": [...]
  }
}
```

#### 2. Planting Recommendations (Comparison)
```
GET /api/recommendations/planting/1
```
**Expected Response:**
```json
{
  "success": true,
  "comparison": {
    "comparison": [
      {
        "crop": "Maize",
        "suitability_score": 85.5,
        ...
      }
    ]
  }
}
```

#### 3. Detailed Crop Recommendation
```
GET /api/recommendations/planting/1?crop_type=Maize
```
**Expected Response:**
```json
{
  "success": true,
  "recommendation": {
    "crop_type": "Maize",
    "overall_recommendation": "highly_recommended",
    "suitability_score": 85.5,
    "planting_window": {...},
    "weather_analysis": {...},
    ...
  }
}
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "Could not validate credentials"
**Solution:**
- Login again to get fresh auth token
- Check localStorage for 'auth_token'

### Issue 2: Weather not loading
**Solution:**
- Check farm has latitude/longitude
- Verify OpenWeatherMap API key in backend/.env
- Check backend logs for API errors

### Issue 3: Planting page shows "Failed to load"
**Solution:**
- Ensure farm exists with valid ID
- Check farm has GPS coordinates
- Check backend logs

### Issue 4: Scores all the same
**Solution:**
- Check if farm has different soil/climate data
- Verify ML model is loaded
- Check backend logs for ML errors

---

## ✨ Expected User Experience

### Good Recommendation Example:
```
Crop: Maize
Score: 85.5/100
Status: Highly Recommended
Summary: "Excellent time to plant (85.5/100). Weather optimal.
         Peak planting season. Soil ideal. ML model: 87% suitable"

Planting Window:
- Recommended: November 5, 2025
- Earliest: November 3, 2025
- Latest: November 10, 2025

Weather:
- Temperature: 24.5°C (optimal)
- Rainfall: 120mm expected (adequate)
- 4 rainy days ahead

Season: Primary planting season (Months: 3, 4, 5)
Soil: pH 6.5 (optimal), Loam soil (suitable)
```

### Marginal Recommendation Example:
```
Crop: Tomato
Score: 55/100
Status: Recommended (with caution)
Summary: "Good time to plant with proper care (55/100).
         Weather acceptable. Acceptable season. Soil acceptable"

Concerns:
- Temperature slightly low for tomatoes
- Consider protection during cooler nights
```

### Not Recommended Example:
```
Crop: Rice
Score: 35/100
Status: Not Recommended
Summary: "Wait for better conditions (35/100). Weather unfavorable.
         Off-season. Soil needs amendment"

Reason: Wait for primary planting season (Months: 4, 5, 6)
```

---

## 📊 Data to Verify

### Farm Requirements for Best Results:
- ✅ GPS coordinates (latitude/longitude)
- ✅ Soil pH value
- ✅ Soil type
- ✅ Climate zone
- ✅ Average temperature
- ✅ Average annual rainfall

**Without this data:**
- Recommendations still work
- Scores may be less accurate
- Some factors will be generic

---

## 🎓 Learning Points

### Scoring Algorithm:
- **Weather**: 40% weight (current + 7-day forecast)
- **Season**: 30% weight (planting calendar)
- **Soil**: 20% weight (pH + type compatibility)
- **ML Model**: 10% weight (if available)

### Recommendation Thresholds:
- **75-100**: Highly Recommended (green)
- **50-74**: Recommended (blue)
- **0-49**: Not Recommended (red)

---

## ✅ Testing Checklist

### Backend Testing:
- [ ] All 5 recommendation endpoints working
- [ ] Weather API returns data
- [ ] Planting service calculates scores correctly
- [ ] Error handling works
- [ ] Mock data fallback works

### Frontend Testing:
- [ ] Weather widget displays on farm detail
- [ ] Planting button navigates correctly
- [ ] Comparison view shows all crops
- [ ] Detailed view loads for each crop
- [ ] All data sections render
- [ ] Loading states work
- [ ] Error states handled
- [ ] Back navigation works

### Integration Testing:
- [ ] Backend + Frontend communication
- [ ] Auth tokens work
- [ ] Data flows correctly
- [ ] Real-time weather updates
- [ ] Recommendations reflect current conditions

---

## 🚀 Next Steps After Testing

1. **If Everything Works:**
   - ✅ Phase 3 is complete!
   - Move to production deployment prep
   - Consider adding harvest predictions UI
   - Add more crops to the system

2. **If Issues Found:**
   - Check browser console for errors
   - Check backend logs
   - Verify database has required farm data
   - Test with different farms

3. **Enhancements to Consider:**
   - Add harvest predictions to crop cards
   - Create weather alerts/notifications
   - Add crop care recommendations
   - Build farmer dashboard with insights

---

## 📝 Report Template

After testing, document your results:

```
PHASE 3 TESTING RESULTS
Date: _____________
Tester: ___________

Backend Status:
- API endpoints: ☐ Pass ☐ Fail
- Weather integration: ☐ Pass ☐ Fail
- Recommendations: ☐ Pass ☐ Fail

Frontend Status:
- Weather widget: ☐ Pass ☐ Fail
- Planting page: ☐ Pass ☐ Fail
- Navigation: ☐ Pass ☐ Fail

Issues Found:
1. _____________________
2. _____________________

Overall Status: ☐ PASS ☐ NEEDS WORK
```

---

**Happy Testing! 🎉**

If you encounter any issues, check:
1. Browser console (F12)
2. Backend terminal logs
3. Network tab (F12 → Network)
4. Database has farms with GPS coordinates
