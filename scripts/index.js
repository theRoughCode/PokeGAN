let model;

const loadingTextArr = [
    'Catching wild Pidgeys',
    'Talking to Professor Oak',
    'Fighting Team Rocket',
    'Fishing up Magikarp',
    'Training with Machoke',
    'Running from the Squirtle Squad',
    'Saying goodbye to Butterfree',
    'Getting ignored by Charmeleon',
    'Choosing a starter Pokémon'
];

// Load model
async function load() {
    const minWaitTime = 400;
    const loadingText = loadingTextArr[Math.floor(Math.random() * loadingTextArr.length)];
    $('#loading-text').html(`${loadingText}&hellip;`);
    try {
        const startTime = new Date().getTime();
        model = await fetchModel();
        while (new Date().getTime() - startTime < minWaitTime) {}
        $('#loading').fadeOut(500);
        setTimeout(() => {
            $('html').css('background-color', 'transparent');
            $('body').css('background-color', 'transparent');
            // $('body').css('background-image', 'url("assets/background.jpg")');
            $('html').css('height', 'auto');
            $('body').css('height', 'auto');
            // $('body').css('padding', '20px');
            $('#main').fadeIn(100);
        }, minWaitTime);
    } catch (error) {
        console.log(error);
    }
}

function generatePokemon() {
    predict(model);
}

load();