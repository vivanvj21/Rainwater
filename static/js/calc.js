$('#calculate').click(function(e) {
    // Constants
    var batteryCapacity = 12; // kWh
    var batteryPrice = 323457; // USD
    var sunPerDay = 4; // hours of sun per day
    var panelGeneration = 0.15; // kWh per panel
    var generatePrice = 0.3; // USD per watt generated

    var bdInKw = 10; // battery discharge in kW (this can be fetched from location-specific data)
    var electricityPrice = 0.12; // USD per kWh

    // Getting values from the user input
    var fans = $('#householdBDs').val();
    var ac = $('#householdBDs1').val();
    var cfl = $('#householdBDs2').val();
    var pc = $('#householdBDs3').val();

    // Calculating consumption for each device
    var consumefan = fans * bdInKw * 0.15;
    var consumeac = ac * bdInKw * 0.85;
    var consumecfl = cfl * bdInKw * 0.20;
    var consumepc = pc * bdInKw * 0.45;

    // Total consumption
    var consumetot = consumefan + consumeac + consumecfl + consumepc;

    // Calculating the number of batteries required
    var batteries = Math.floor(consumetot / batteryCapacity);
    $('#batteryPacks').val(batteries);

    // Calculating the required panel size (capacity)
    var generateCapacity = consumetot / sunPerDay;
    var panelSize = generateCapacity / panelGeneration;
    $('#panelSize').val(panelSize);

    // Calculating the total investment
    var investment = (batteries * batteryPrice) + (generateCapacity * generatePrice);
    $('#investment').val(investment);

    // Calculating return on investment (ROI)
    var returnOfInvestment = Math.floor(investment / electricityPrice / 24 / 365);
    returnOfInvestment = Math.round(returnOfInvestment / 77); // Adjusted for the factor of 77
    $('#returnOfInvestment').val(returnOfInvestment + ' year(s)');
});
