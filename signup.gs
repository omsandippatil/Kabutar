function addDataToSheet(email, password, displayname, handle, bio, biolink) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  
  // Get the last row number
  var lastRow = sheet.getLastRow();
  
  // Add data to the next row
  sheet.getRange(lastRow + 1, 1).setValue(lastRow); //using it for user id
  sheet.getRange(lastRow + 1, 2).setValue(email);
  sheet.getRange(lastRow + 1, 3).setValue(password);
  sheet.getRange(lastRow + 1, 4).setValue(displayname);
  sheet.getRange(lastRow + 1, 5).setValue(handle);
  sheet.getRange(lastRow + 1, 6).setValue(bio);
  sheet.getRange(lastRow + 1, 7).setValue(biolink);
}

// Google Apps Script web app endpoint
function doGet(e) {
  var email = e.parameter.email;
  var password = e.parameter.password;
  var displayname = e.parameter.displayname;
  var handle = e.parameter.handle;
  var bio = e.parameter.bio;
  var biolink= e.parameter.biolink;

  // Call the addDataToSheet function with the provided parameters
  var result = addDataToSheet(email, password, displayname, handle, bio, biolink);

  // Return the result as JSON
  return ContentService.createTextOutput(JSON.stringify(result)).setMimeType(ContentService.MimeType.JSON);
}
