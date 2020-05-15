let model, input;

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

function generateLinkedImage() {
    const urlParams = new URLSearchParams(window.location.search);
    if (!urlParams.has('str')) return;
    const str = urlParams.get('str');
    input = decodeStr(str);
    predict(model, input);
}

// Load model
async function load() {
    const minWaitTime = 400;
    const loadingText = loadingTextArr[Math.floor(Math.random() * loadingTextArr.length)];
    $('#loading-text').html(`${loadingText}&hellip;`);
    try {
        const startTime = new Date().getTime();
        model = await fetchModel();
        // Generate image if linked
        generateLinkedImage();
        while (new Date().getTime() - startTime < minWaitTime) {}
        $('#loading').fadeOut(500);
        setTimeout(() => {
            $('html').css('background-color', 'transparent');
            $('body').css('background-color', 'transparent');
            $('html').css('height', 'auto');
            $('body').css('height', 'auto');
            $('#main').fadeIn(100);
        }, minWaitTime);
    } catch (error) {
        console.log(error);
    }
}

function generatePokemon() {
    input = predict(model);
    return input;
}

function sharePage() {
    if (!navigator.share) {
        return;
    }
    navigator.share({
        title: 'Pokémon Generator',
        text: 'Generate your own Pokémon!',
        url: 'https://www.raphaelkoh.me/PokeGAN/',
    });
}

async function share(name) {
    const { dataURL, width, height } = await generateShareImage(name);
    const imgFile = dataURLtoFile(dataURL, `${name}.png`);
    if (!navigator.canShare || !navigator.canShare({ files: [imgFile] })) {
        openImageInNewTab(dataURL, width, height);
        return;
    }
    navigator.share({
        files: [imgFile],
        title: name,
        text: 'Generate your own Pokémon!',
        url: 'https://www.raphaelkoh.me/PokeGAN/'
    }).catch((err) => console.log(err));
    // // TODO: Share image
    // const title = 'Pokémon Generator';
    // const text = 'Generate your own Pokémon!'
    // let url = 'https://www.raphaelkoh.me/PokeGAN/';
    // if (input != null) {
    //     // Convert tensor to encoded string
    //     const str = encodeTensor(input);
    //     url += '?str=' + str;
    // }
    // navigator.share({
    //     title,
    //     text,
    //     url
    // });
}

load();